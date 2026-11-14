"""Local job-list loading and stable identities. Does not open job websites."""
import hashlib
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

DEFAULT_LINKS = Path(__file__).with_name('elton_fly_job_links.txt')


def canonical_url(url):
    parts = urlsplit(url.strip())
    if parts.scheme not in ('http', 'https') or not parts.hostname or parts.username or parts.password:
        raise ValueError('job links must be HTTP(S) URLs without credentials')
    # Preserve job identifiers; discard only known tracking parameters.
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
             if not k.lower().startswith('utm_') and k.lower() not in ('fbclid', 'gclid')]
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip('/') or '/',
                       urlencode(sorted(query)), ''))


def job_id(job):
    if job.get('url'):
        identity = canonical_url(job['url'])
    else:
        # Explicit injected fixture records need stable identities too.
        identity = str(job.get('id') or job.get('title', 'fixture'))
    return hashlib.sha256(identity.encode()).hexdigest()


def load_jobs(path=DEFAULT_LINKS):
    jobs, seen = [], set()
    for number, raw in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines(), 1):
        raw = raw.strip()
        if not raw or raw.startswith('#'):
            continue
        try:
            url = canonical_url(raw)
        except ValueError as exc:
            raise ValueError(f'{path}:{number}: {exc}') from exc
        if url not in seen:
            # A URL provides no evidence of title or relevance. Do not invent it.
            jobs.append({'url': url, 'title': f'Job {len(jobs) + 1}', 'skills': []})
            seen.add(url)
    if not jobs:
        raise ValueError('job list is empty')
    return jobs
