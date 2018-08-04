import openpyxl

wb = openpyxl.load_workbook(filename='8OHDG RUN 3 21.06.18.xlsx', data_only=True)
sheet = wb["End point_1"]

tmp = [sheet.cell(row=row, column=col).value for col in range(2, 14) for row in range(31, 35)]

STANDARD_DATA = tmp[1:8]

DATA = tmp[9:]
