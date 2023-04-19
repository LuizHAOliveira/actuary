import xlwings as xw
from .triangle import Triangle

def triangle_to_excel(triangle: Triangle, wb: xw.Book = None, ws: xw.Sheet = None) -> xw.Sheet:
        if not ws and not wb:
            wb = xw.Book()
            ws = wb.sheets.add()
        elif not ws:
            ws = wb.sheets.add()
        # ws.range('D1').value = self.h_header.h_numbers
        # ws.range('A4').value = self.v_header.h_dates
        ws.range('D4').value = triangle.values
        return ws