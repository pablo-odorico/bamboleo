from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, cm
from reportlab.lib.colors import red, black
from reportlab.graphics import barcode, renderPDF
from reportlab.graphics.shapes import Drawing

def drawQR(c, text, size, x, y):
    qrCode = barcode.qr.QrCodeWidget(text, barLevel='M', qrVersion=2, barBorder=1)
    bounds = qrCode.getBounds()
    drawing = Drawing(size, size, transform=[size / (bounds[2] - bounds[0]), 0, 0, size / (bounds[3] - bounds[1]), 0, 0])
    drawing.add(qrCode)
    renderPDF.draw(drawing, c, x - size/2, y - size/2)

def drawLabel(c, leftLabel, rightLabel, width=10*cm, height=3*cm):
    c.saveState()

    def vertLine(x, yPercent0, yPercent1):
        c.line(x, yPercent0 * height, x, yPercent1 * height)

    # 4 vertical folding lines for the label that stays in the cut
    c.setStrokeGray(0.6)
    c.setLineCap(1)
    c.setDash(2, 3)
    vertLine(1/5 * width, 1/2, 1)
    vertLine(2/5 * width, 1/2, 1)
    vertLine(3/5 * width, 1/2, 1)
    vertLine(4/5 * width, 1/2, 1)
    c.setDash(1)
    c.setLineCap(0)
    c.setStrokeGray(1)

    # 2 vertical double cut lines in red for the saw
    c.setStrokeColor(red)
    c.setLineWidth(2)
    cutClearence = 2.5*mm
    cutStart = 2/5 * width - cutClearence
    cutEnd = 3/5 * width + cutClearence
    vertLine(cutStart, 0, 1/2)
    vertLine(cutEnd, 0, 1/2)
    c.setLineWidth(1)
    c.setStrokeColor(black)

    # 2 horizontal cut lines for scissors
    c.setDash(6, 3)
    c.line(0/5 * width, height/2, 2/5 * width, height/2)
    c.line(3/5 * width, height/2, 5/5 * width, height/2)
    c.setDash(1)

    # QR codes
    qrSize = height/2 - 2*mm
    if leftLabel:
        drawQR(c, leftLabel, x=1/5 * width - qrSize/2 - 1*mm, y=3/4 * height, size=qrSize)
        drawQR(c, leftLabel, x=0/5 * width + qrSize/2 + 1*mm, y=1/4 * height, size=qrSize)
    if rightLabel:
        drawQR(c, rightLabel, x=4/5 * width + qrSize/2 + 1*mm, y=3/4 * height, size=qrSize)
        drawQR(c, rightLabel, x=5/5 * width - qrSize/2 - 1*mm, y=1/4 * height, size=qrSize)

    txt = c.beginText()
    txt.setTextOrigin(0, 0)
    txt.textLine('Tube 1')
    txt.textLine('Section 2B')
    c.drawText(txt)

    c.rect(0, 0, width, height)
    c.restoreState()


c = canvas.Canvas('labels.pdf', pagesize=A4)
c.translate(5*cm, 10*cm)
c.setTitle('Bamboleo Labels')

drawLabel(c, leftLabel='11AL', rightLabel='11AR')
c.save()
