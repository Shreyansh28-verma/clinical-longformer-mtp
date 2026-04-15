import markdown
from xhtml2pdf import pisa
import os

def convert_md_to_pdf(md_in, pdf_out, css_in):
    # Read markdown and CSS
    with open(md_in, 'r', encoding='utf-8') as f:
        md_text = f.read()
    
    with open(css_in, 'r', encoding='utf-8') as f:
        css_text = f.read()

    # Convert markdown to html (with extensions for tables, fenced code, etc.)
    html_body = markdown.markdown(md_text, extensions=['extra', 'tables', 'fenced_code'])

    # wrap in simple HTML structure with CSS
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
    @page {{ size: a4 portrait; margin: 2cm; }}
    {css_text}
    </style>
    </head>
    <body>
    {html_body}
    </body>
    </html>
    """

    # Generate PDF
    with open(pdf_out, "wb") as result_file:
        pisa_status = pisa.CreatePDF(
            src=html_content, dest=result_file)

    if pisa_status.err:
        print(f"Error during PDF generation: {pisa_status.err}")
    else:
        print("PDF generated successfully:", pdf_out)

if __name__ == "__main__":
    md_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_report.md")
    convert_md_to_pdf(md_file, "results_report.pdf", "report_style.css")
