import argparse
import os

from PyPDF2 import PdfFileReader, PdfFileWriter
import sys

# def split(path, name_of_split):
#     pdf = PdfFileReader(path)
#     pdf_writer = PdfFileWriter()
#     for page in range(pdf.getNumPages()):
#         pdf_writer.addPage(pdf.getPage(page))
#         if (page+1) %5 == 0:
#             output = f'{name_of_split}{page}.pdf'
#             with open(output, 'wb') as output_pdf:
#                 pdf_writer.write(output_pdf)
#             pdf_writer = PdfFileWriter()

def split(path, output_path, perPages):
    pdf_dir = path[:path.rfind('\\')]
    pdf_name = path[path.rfind('\\')+1:-4]
    pdf = PdfFileReader(path)
    pdf_writer = PdfFileWriter()
    last_page = 1
    if not output_path:
        output_path = pdf_dir
    output_path += "/" + pdf_name
    for page in range(pdf.getNumPages()):
        pdf_writer.addPage(pdf.getPage(page))
        if (page + 1) % perPages == 0 or page == pdf.getNumPages() -1:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            output = "%s/%d-%d.pdf" % (output_path, last_page, page+1)
            print(output)
            last_page = page +1
            with open(output, 'wb') as output_pdf:
                pdf_writer.write(output_pdf)
                pdf_writer = PdfFileWriter()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, dest="path", default=r'C:\SameLin\SCU\318__论文\Paper\可解释性可视化\Visual Interaction with Deep Learning Models through Collaborative Semantic Inference.pdf')
    parser.add_argument("--output", type=str, dest="outputPath")
    parser.add_argument("--pages", type=int, dest="pages", default=4)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    split(args.path, args.outputPath, args.pages)