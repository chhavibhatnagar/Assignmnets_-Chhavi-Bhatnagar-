import fitz   
import re


def _normalize(text):
    """Collapse common formatting differences that cause exact-match misses."""
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _highlight_text_on_all_pages(doc, text):
    """Search every page for `text` and highlight all matches. Returns True if found."""
    found = False
    for page in doc:
        rects = page.search_for(text)
        for rect in rects:
            annot = page.add_highlight_annot(rect)
            annot.update()
            found = True
    return found


def highlight_pdf(input_pdf, headings, output_pdf):
    """
    Open input_pdf, search for each heading's text on every page, and
    highlight all matches. Tries an exact match first; if that fails,
    retries with whitespace/punctuation normalized (handles curly quotes,
    en/em dashes, and stray double-spaces that break exact matching).
    Saves result to output_pdf. Returns the number of headings matched.
    """
    doc = fitz.open(input_pdf)
    matched = 0

    for heading in headings:
        heading = heading.strip()
        if not heading:
            continue

        found = _highlight_text_on_all_pages(doc, heading)

        if not found:
            normalized = _normalize(heading)
            if normalized != heading:
                found = _highlight_text_on_all_pages(doc, normalized)

        if not found:
            words = heading.split()
            if len(words) > 3:
                partial = " ".join(words[:4])
                found = _highlight_text_on_all_pages(doc, partial)

        if found:
            matched += 1

    doc.save(output_pdf)
    doc.close()
    return matched