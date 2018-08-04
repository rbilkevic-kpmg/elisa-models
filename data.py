import openpyxl


class ElisaData:
    def __init__(self, elisa_data):
        self.elisa_data = elisa_data
        self._process()

    def _process(self):
        self._wb = openpyxl.load_workbook(filename=self.elisa_data, data_only=True)
        self._sheet = self._wb["End point_1"]

        tmp = [self._sheet.cell(row=row, column=col).value for col in range(2, 14) for row in range(31, 35)]

        self.STANDARDS_DATA = tmp[1:8]
        self.DATA = tmp[9:]

    def write_concentrations(self, conc_data):
        conc_data = 9 * [None] + list(conc_data)
        for col in range(2, 14):
            for row in range(38, 42):
                self._sheet.cell(row=row, column=col).value = conc_data.pop(0)

        self._wb.save(self.elisa_data.replace('input\\', 'output\\').replace(".xlsx", "_result.xlsx"))
        print("Output saved in output folder")
