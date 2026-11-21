"""Generate small DOCX and PDF fixtures (no binary files checked in by hand)."""

from pathlib import Path


def write_docx(path: Path, title: str, paragraphs: list[str]) -> None:
    import docx

    d = docx.Document()
    d.core_properties.title = title
    d.add_heading(title, level=1)
    for p in paragraphs:
        d.add_paragraph(p)
    d.save(str(path))


def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_pdf(path: Path, pages: list[list[str]]) -> None:
    """Minimal valid PDF with one text line per list item."""
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", None,
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for lines in pages:
        ops = ["BT /F1 11 Tf 14 TL 72 740 Td"] + [f"({_escape(l)}) Tj T*" for l in lines] + ["ET"]
        stream = "\n".join(ops).encode("latin-1")
        objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
        content_id = len(objs)
        objs.append(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents %d 0 R "
                    b"/Resources << /Font << /F1 3 0 R >> >> >>" % content_id)
        kids.append(len(objs))
    objs[1] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (
        b" ".join(b"%d 0 R" % k for k in kids), len(kids))
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    path.write_bytes(bytes(out))


def write_samples(folder: Path) -> None:
    write_docx(folder / "energy-transition.docx", "Energy Transition Notes", [
        "The International Energy Agency reported that renewable capacity grew by 50 percent in 2023, "
        "led by solar installations in China.",
        "Grid operators in Germany and California are adding battery storage to manage the variability "
        "of renewable generation. Tesla and CATL are among the largest suppliers of grid-scale batteries.",
        "Some analysts compare the energy consumption of proof-of-work cryptocurrencies such as Bitcoin "
        "with that of small countries, arguing that mining competes with other uses for clean electricity.",
    ])
    write_pdf(folder / "research" / "flood-policy.pdf", [
        ["Flood Policy Brief", "",
         "The Dutch Delta Programme coordinates flood protection across the Netherlands.",
         "Rotterdam and Amsterdam share funding for dikes, dunes and storm surge barriers."],
        ["The European Environment Agency expects coastal flood damage to rise sharply by 2100.",
         "Jakarta and Miami are often cited as cities where sea level rise and subsidence combine."],
    ])


if __name__ == "__main__":
    import sys

    write_samples(Path(sys.argv[1]))
