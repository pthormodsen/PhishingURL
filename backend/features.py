from urllib.parse import urlsplit


def url_to_model_text(url):
    url = url.strip()
    parsed = urlsplit(url if "://" in url else "https://" + url)

    hostname = (parsed.hostname or "").lower()

    if hostname.startswith("www."):
        hostname = hostname[4:]

    text = hostname + parsed.path

    if parsed.query:
        text += "?" + parsed.query

    if parsed.fragment:
        text += "#" + parsed.fragment

    return text