import xlsxwriter
import pandas as pd


def save_xlsx(df, csv_time):
    """Formats the excel export file to allow for a nice concise and legible file"""
    name = '/root/WebSentimentWebsite/static/csv_outputs/output' + csv_time + '.xlsx'
    with pd.ExcelWriter(name, engine='xlsxwriter') as writer:
        # remove the index by setting the kwarg 'index' to False
        df.to_excel(excel_writer=writer, sheet_name='output', index=False)

        workbook = writer.book
        worksheet = writer.sheets['output']

        # dynamically set column width
        for i, col in enumerate(df.columns):
            if i == 2:
                column_len = 20
                worksheet.set_column(i, i, column_len)
                continue
            if i == 4:
                column_len = 100
                worksheet.set_column(i, i, column_len)
                continue
            column_len = max(df[col].astype(str).str.len().max(), len(col) + 2)
            worksheet.set_column(i, i, column_len)

        # wrap the text in all cells
        wrap_format = workbook.add_format({'text_wrap': True, 'align': 'center'})
        align_left = workbook.add_format({'text_wrap': True, 'align': 'left'})
        worksheet.set_column(1, len(df.columns) - 2, cell_format=wrap_format)
        worksheet.set_column(len(df.columns)-2, len(df.columns)-1, cell_format=align_left)
        # mimic the default pandas header format for use later
        hdr_fmt = workbook.add_format({
            'bold': True,
            'border': 1,
            'text_wrap': True,
            'align': 'center'
        })

        def update_format(curr_frmt, new_prprty, wrkbk):
            """
            Update a cell's existing format with new properties
            """
            new_frmt = curr_frmt.__dict__.copy()

            for k, v in new_prprty.items():
                new_frmt[k] = v

            new_frmt = {
                k: v
                for k, v in new_frmt.items()
                if (v != 0) and (v is not None) and (v != {}) and (k != 'escapes')
            }

            return wrkbk.add_format(new_frmt)

        # create new border formats
        header_right_thick = update_format(hdr_fmt, {'right': 2}, workbook)
        normal_right_thick = update_format(wrap_format, {'right': 2}, workbook)
        normal_bottom_thick = update_format(wrap_format, {'bottom': 2}, workbook)
        normal_corner_thick = update_format(wrap_format, {
            'right': 2,
            'bottom': 2
        }, workbook)

        # list the 0-based indices where you want bold vertical border lines
        vert_indices = [4]

        # create vertical bold border lines
        for i in vert_indices:
            # header vertical bold line
            worksheet.conditional_format(0, i, 0, i, {
                'type': 'formula',
                'criteria': 'True',
                'format': header_right_thick
            })
            # body vertical bold line
            worksheet.conditional_format(1, i,
                                         len(df.index) - 1, i, {
                                             'type': 'formula',
                                             'criteria': 'True',
                                             'format': normal_right_thick
                                         })
            # bottom corner bold lines
            worksheet.conditional_format(len(df.index), i, len(df.index), i, {
                'type': 'formula',
                'criteria': 'True',
                'format': normal_corner_thick
            })
        # create bottom bold border line
        for i in [i for i in range(len(df.columns) - 1) if i not in vert_indices]:
            worksheet.conditional_format(len(df.index), i, len(df.index), i, {
                'type': 'formula',
                'criteria': 'True',
                'format': normal_bottom_thick
            })
