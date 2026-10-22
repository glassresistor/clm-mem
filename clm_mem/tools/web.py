"""Web tools: search providers and page fetch, pluggable by provider.

Providers are chosen by ``search_provider`` in the harness config:

* ``stub``     -- deterministic offline results (used by tests / ``--use-mock``)
* ``ddg``      -- DuckDuckGo HTML endpoint, no API key
* ``brave``    -- Brave Search API ($BRAVE_API_KEY)
* ``tavily``   -- Tavily search API ($TAVILY_API_KEY)
* ``exa``      -- Exa search API ($EXA_API_KEY)
* ``off``      -- no web access at all

Every provider returns the same shape: a list of ``{title, url, snippet}`` plus
a ``provider`` note. Fetching strips HTML into readable text and caps length,
so what lands in the memory window is bounded and predictable.
"""
from __future__ import annotations

import html
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List, Optional

UA = "clm-mem/0.4 (+https://github.com/unsloth/clm-mem) research client"
BLOCKED_HOSTS = {"169.254.169.254", "metadata.google.internal",
                 "metadata.http.ibmcloud.com"}

_TAG_RE = re.compile(r"(?s)<(script|style|noscript)[^>]*>.*?</\1>", re.I)
_A_TAG = re.compile(r"<a\b[^>]*>", re.I)
_ANY_TAG = re.compile(r"<[^>]{1,400}>")
_DDG_RESULT = re.compile(
    r'<a[^>]+class="result__a"[^>]*href="(?P<href>[^"]*)"[^>]*>(?P<ti>.*?)</a>'
    r'(?:.*?)(?:<td class="result-snippet">|'
    r'<div class="snippet__second_line">)'
    r'(?P<sn>.*?)(?:</td>|</div>)', re.I | re.S)


def _clean_ddg_href(href: str) -> str:
    """DuckDuckGo wraps result links in a redirect (``/l/?uddg=...``)."""
    for prefix in ("//duckduckgo.com/l/?uddg=", "//duckduckgo.com/redirect"):
        if prefix in href:
            q = urllib.parse.parse_qs(urllib.parse.urlparse(
                "https:" + href if href.startswith("//") else href
            ).query.get("uddg", [""]))
            vals = q.get("uddg") or [""]
            return urllib.parse.unquote(vals[0]) if vals[0] else href
    if href.startswith("//"):
        return "https:" + href
    return href


def _norm_url(u: str) -> str:
    if not u:
        return ""
    if u.startswith("//"):
        return "https:" + u
    return u


def strip_html(raw: str, limit: int = 6000) -> str:
    """Crude but cheap HTML -> readable text (no extra dependency)."""
    text = _TAG_RE.sub("", raw)
    text = re.sub(r"<br[^>]*>", "\n", text, flags=re.I)
    text = re.sub(r"</(p|div|li|h[1-6]|tr)[^>]*>", "\n", text, flags=re.I)
    text = _A_TAG.sub("", text)
    text = _ANY_TAG.sub("", text)
    text = html.unescape(text)
    lines = [ln.strip() for ln in text.splitlines()]
    out = "\n".join(ln for ln in lines if ln)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    if len(out) > limit:
        out = out[:limit] + f"...[truncated {len(out) - limit} chars]"
    return out


class WebError(RuntimeError):
    pass


def _get(url: str, headers: Optional[Dict[str, str]] = None,
         timeout: float = 15.0, max_bytes: int = 400_000) -> str:
    host = urllib.parse.urlparse(url).hostname or ""
    if host in BLOCKED_HOSTS:
        raise WebError(f"refused fetch of blocked host {host}")
    req = urllib.request.Request(url, headers={"User-Agent": UA,
                                              "Accept-Language": "en",
                                              **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(max_bytes).decode(errors="replace")


# ---------------------------------------------------------------- providers

def search_stub(query: str, limit: int = 5) -> List[Dict[str, str]]:
    """Offline deterministic results (tests / mock mode / air-gapped boxes)."""
    slug = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-") or "query"
    out = []
    for i in range(max(1, min(limit, 5))):
        out.append({"title": f"{query} — reference {i + 1}",
                    "url": f"https://example.org/{slug}/{i + 1}",
                    "snippet": f"Offline stub snippet {i + 1} for “{query}”. "
                               f"Set --search-provider ddg (or an API-keyed "
                               f"provider) for real results."})
    return out


def search_ddg(query: str, limit: int = 5) -> List[Dict[str, str]]:
    body = urllib.parse.urlencode({"q": query, "kl": "us-en"})
    raw = _get(f"https://html.duckduckgo.com/html/?{body}")
    hits = []
    for m in _DDG_RESULT.finditer(raw):
        href = _clean_ddg_href(m.group("href"))
        if not href.startswith("http"):
            continue
        snippet = html.unescape(_ANY_TAG.sub("", m.group("sn"))).strip()
        hits.append({"title": html.unescape(_ANY_TAG.sub("", m.group("ti"))).strip(),
                     "url": href, "snippet": snippet})
        if len(hits) >= limit:
            break
    return hits


def search_brave(query: str, limit: int = 5) -> List[Dict[str, str]]:
    key = os.environ.get("BRAVE_API_KEY", "")
    if not key:
        raise WebError("brave provider needs BRAVE_API_KEY")
    body = urllib.parse.urlencode({"q": query, "count": max(1, min(limit, 10))})
    raw = _get(f"https://api.search.brave.com/res/v1/web/search?{body}",
               {"Accept": "application/json", "Accept-Encoding": "gzip",
                "X-Subscription-Token": key})
    data = json.loads(raw)
    hits = [{"title": r.get("meta_title", "") or r.get("title", ""),
             "url": _norm_url(r.get("url", "")),
             "snippet": html.unescape(r.get("snippet", ""))}
            for r in (data.get("results") or data.get("web", {})
                      .get("results", []))]
    return hits[:limit]


def search_tavily(query: str, limit: int = 5) -> List[Dict[str, str]]:
    key = os.environ.get("TAVILY_API_KEY", "")
    if not key:
        raise WebError("tavily provider needs TAVILY_API_KEY")
    req = urllib.request.Request(
        "https://api.tavily.com/search",
        data=json.dumps({"query": query, "max_results": limit}).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.loads(r.read().decode())
    return [{"title": d.get("title", ""), "url": d.get("url", ""),
             "snippet": (d.get("content") or "")[:400]} for d in data.get(
                 "results", [])][:limit]


def search_exa(query: str, limit: int = 5) -> List[Dict[str, str]]:
    key = os.environ.get("EXA_API_KEY", "")
    if not key:
        raise WebError("exa provider needs EXA_API_KEY")
    req = urllib.request.Request(
        "https://api.exa.ai/search",
        data=json.dumps({"query": query, "numResults": limit}).encode(),
        headers={"Content-Type": "application/json", "x-api-key": key})
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.loads(r.read().decode())
    return [{"title": d.get("title", ""), "url": d.get("url", ""),
             "snippet": (d.get("text") or "")[:400]} for d in data.get(
                 "results", [])][:limit]


PROVIDERS: Dict[str, object] = {
    "stub": search_stub,
    "ddg": search_ddg,
    "brave": search_brave,
    "tavily": search_tavily,
    "exa": search_exa,
}

#: providers that work without any API key
KEYLESS = ("stub", "ddg")


def run_search(provider: str, query: str, limit: int = 5) -> List[Dict[str, str]]:
    """Dispatch to a provider; ``off`` raises, unknown names fall back to ddg."""
    if provider in ("", "off", None):
        raise WebError("web access is disabled for this session")
    fn = PROVIDERS.get(provider)
    if fn is None:
        fn = PROVIDERS["ddg"]
    return fn(query, limit)


def fetch_page(url: str, limit: int = 6000) -> Dict[str, object]:
    """Fetch one page as readable text (title + body)."""
    if "://" not in url:
        url = "https://" + url
    raw = _get(url, max_bytes=min(400_000, limit * 40))
    title = ""
    m = re.search(r"<title[^>]*>(.*?)</title>", raw, re.I | re.S)
    if m:
        title = html.unescape(_ANY_TAG.sub("", m.group(1))).strip()
    body = strip_html(raw, limit)
    return {"url": url, "title": title, "text": body,
            "chars": len(body), "status": "ok"}
