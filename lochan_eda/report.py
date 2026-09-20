import numpy as np

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT


class Report:
    def __init__(self, profiler):
        self.profiler = profiler

    def save(self, filepath="report.pdf"):
        """Generate EDA PDF report."""

        # ---------------------------------------------------------
        # Color system
        # ---------------------------------------------------------
        NAVY = colors.HexColor("#183B56")
        BLUE = colors.HexColor("#2F6690")
        TEXT = colors.HexColor("#243B53")
        MUTED = colors.HexColor("#627D98")
        BORDER = colors.HexColor("#D9E2EC")
        LIGHT = colors.HexColor("#F5F7FA")
        HEADER_BG = colors.HexColor("#EAF1F7")
        WHITE = colors.white

        # ---------------------------------------------------------
        # Page header / footer
        # ---------------------------------------------------------
        def add_metadata(canvas, doc):
            canvas.saveState()

            canvas.setTitle("Data Exploratory Report")
            canvas.setAuthor("Lochan Jangid")
            canvas.setSubject("Lochan EDA")

            width, height = A4

            # Top accent
            canvas.setFillColor(NAVY)
            canvas.rect(
                0,
                height - 6,
                width,
                6,
                stroke=0,
                fill=1,
            )

            # Header
            canvas.setFont("Helvetica-Bold", 9)
            canvas.setFillColor(NAVY)
            canvas.drawString(42, height - 30, "DATA EXPLORATORY REPORT")

            canvas.setFont("Helvetica", 7)
            canvas.setFillColor(MUTED)
            canvas.drawRightString(
                width - 42,
                height - 30,
                "Lochan EDA",
            )

            # Header divider
            canvas.setStrokeColor(BORDER)
            canvas.setLineWidth(0.6)
            canvas.line(
                42,
                height - 38,
                width - 42,
                height - 38,
            )

            # Footer
            canvas.setStrokeColor(BORDER)
            canvas.line(
                42,
                32,
                width - 42,
                32,
            )

            canvas.setFont("Helvetica", 7)
            canvas.setFillColor(MUTED)

            canvas.drawString(
                42,
                20,
                "Exploratory Data Analysis",
            )

            canvas.drawRightString(
                width - 42,
                20,
                f"Page {doc.page}",
            )

            canvas.restoreState()

        # ---------------------------------------------------------
        # Document
        # ---------------------------------------------------------
        doc = SimpleDocTemplate(
            filepath,
            pagesize=A4,
            rightMargin=42,
            leftMargin=42,
            topMargin=58,
            bottomMargin=45,
        )

        # ---------------------------------------------------------
        # Styles
        # ---------------------------------------------------------
        styles = getSampleStyleSheet()

        section_style = ParagraphStyle(
            "Section",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=NAVY,
            spaceAfter=8,
        )

        subtitle_style = ParagraphStyle(
            "Subtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=MUTED,
            spaceAfter=10,
        )

        story = []

        # =========================================================
        # PAGE 1
        # =========================================================

        story.append(Spacer(1, 20))

        # Report title
        story.append(
            Paragraph(
                "Data Overview",
                ParagraphStyle(
                    "Title",
                    parent=styles["Heading1"],
                    fontName="Helvetica-Bold",
                    fontSize=22,
                    leading=25,
                    textColor=NAVY,
                    spaceAfter=4,
                ),
            )
        )

        story.append(
            Paragraph(
                "Dataset structure and quality overview",
                subtitle_style,
            )
        )

        # ---------------------------------------------------------
        # Overview table
        # ---------------------------------------------------------
        overview = self.profiler.overview()

        overview["duplicate_percentage"] = (
            f'{overview["duplicate_percentage"]:.2f}%'
        )

        overview["missing_percentage"] = (
            f'{overview["missing_percentage"]:.2f}%'
        )

        overview_table = [
            [k.replace("_", " ").title(), v]
            for k, v in overview.items()
        ]

        overview_tb = Table(
            overview_table,
            colWidths=[175, 280],
            hAlign="LEFT",
        )

        overview_tb.setStyle(
            TableStyle([
                # Labels
                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    NAVY,
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (0, -1),
                    WHITE,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (0, -1),
                    "Helvetica-Bold",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (0, -1),
                    8,
                ),

                # Values
                (
                    "BACKGROUND",
                    (1, 0),
                    (1, -1),
                    LIGHT,
                ),
                (
                    "TEXTCOLOR",
                    (1, 0),
                    (1, -1),
                    TEXT,
                ),
                (
                    "FONTNAME",
                    (1, 0),
                    (1, -1),
                    "Helvetica",
                ),
                (
                    "FONTSIZE",
                    (1, 0),
                    (1, -1),
                    8,
                ),

                # Spacing
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    8,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    8,
                ),

                # Structure
                (
                    "LINEBELOW",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    BORDER,
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
            ])
        )

        story.append(overview_tb)

        # =========================================================
        # PAGE 2
        # =========================================================

        story.append(PageBreak())

        story.append(
            Paragraph(
                "Numerical Analysis",
                section_style,
            )
        )

        story.append(
            Paragraph(
                "Summary statistics for numerical features",
                subtitle_style,
            )
        )

        # ---------------------------------------------------------
        # Numerical Summary
        # ---------------------------------------------------------
        numerical_summary = self.profiler.numerical.summary()

        num_summary_table = [
            ["Columns"] + numerical_summary.columns.tolist()
        ]

        num_summary_table.extend(
            [
                [idx] + list(np.round(row, 2))
                for idx, row in zip(
                    numerical_summary.index,
                    numerical_summary.values,
                )
            ]
        )

        num_summary_tb = Table(
            num_summary_table,
            repeatRows=1,
            hAlign="LEFT",
        )

        num_summary_tb.setStyle(
            TableStyle([
                # Header
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    NAVY,
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    WHITE,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, 0),
                    7,
                ),

                # Body
                (
                    "FONTNAME",
                    (0, 1),
                    (-1, -1),
                    "Helvetica",
                ),
                (
                    "FONTSIZE",
                    (0, 1),
                    (-1, -1),
                    7,
                ),
                (
                    "TEXTCOLOR",
                    (0, 1),
                    (-1, -1),
                    TEXT,
                ),

                # Alternating rows
                *[
                    (
                        "BACKGROUND",
                        (0, r),
                        (-1, r),
                        LIGHT,
                    )
                    for r in range(
                        2,
                        len(num_summary_table),
                        2,
                    )
                ],

                # Alignment
                (
                    "ALIGN",
                    (1, 1),
                    (-1, -1),
                    "RIGHT",
                ),
                (
                    "ALIGN",
                    (0, 0),
                    (0, -1),
                    "LEFT",
                ),
                (
                    "ALIGN",
                    (1, 0),
                    (-1, 0),
                    "CENTER",
                ),

                # Compact spacing
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    3,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    3,
                ),

                # Borders
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.3,
                    BORDER,
                ),
                (
                    "LINEBELOW",
                    (0, 0),
                    (-1, 0),
                    0.8,
                    BLUE,
                ),

                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
            ])
        )

        story.append(num_summary_tb)

        # =========================================================
        # PAGE 3
        # =========================================================

        story.append(PageBreak())

        # ---------------------------------------------------------
        # Numerical plots
        # ---------------------------------------------------------

        story.append(
            Paragraph(
                "Distribution Analysis",
                section_style,
            )
        )

        story.append(
            Paragraph(
                "Distribution, spread, and normality diagnostics",
                subtitle_style,
            )
        )

        numerical_fig = self.profiler.numerical.plot()

        num_fig_img = Image("numerical_plots.png")

        num_fig_img.drawWidth = 400
        num_fig_img.drawHeight = 650

        story.append(num_fig_img)

        # =========================================================
        # PAGE 4
        # =========================================================

        story.append(PageBreak())

        story.append(
            Paragraph(
                "Categorical Summary",
                section_style,
            )
        )

        story.append(
            Paragraph(
                "Summary statistics for categorical features",
                subtitle_style,
            )
        )

        # ---------------------------------------------------------
        # Categorical Summary
        # ---------------------------------------------------------
        categorical_summary = self.profiler.categorical.summary()

        cat_summary_table = [
            ["Columns"] + categorical_summary.columns.tolist()
        ]

        cat_summary_table.extend(
            [
                [idx] + list(row)
                for idx, row in zip(
                    categorical_summary.index,
                    categorical_summary.values,
                )
            ]
        )

        cat_summary_tb = Table(
            cat_summary_table,
            repeatRows=1,
            hAlign="LEFT",
        )

        cat_summary_tb.setStyle(
            TableStyle([
                # Header
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    NAVY,
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    WHITE,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, 0),
                    7,
                ),

                # Body
                (
                    "FONTNAME",
                    (0, 1),
                    (-1, -1),
                    "Helvetica",
                ),
                (
                    "FONTSIZE",
                    (0, 1),
                    (-1, -1),
                    7,
                ),
                (
                    "TEXTCOLOR",
                    (0, 1),
                    (-1, -1),
                    TEXT,
                ),

                # Alternating rows
                *[
                    (
                        "BACKGROUND",
                        (0, r),
                        (-1, r),
                        LIGHT,
                    )
                    for r in range(
                        2,
                        len(cat_summary_table),
                        2,
                    )
                ],

                # Alignment
                (
                    "ALIGN",
                    (1, 1),
                    (-1, -1),
                    "RIGHT",
                ),
                (
                    "ALIGN",
                    (0, 0),
                    (0, -1),
                    "LEFT",
                ),
                (
                    "ALIGN",
                    (1, 0),
                    (-1, 0),
                    "CENTER",
                ),

                # Compact spacing
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    3,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    3,
                ),

                # Borders
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.3,
                    BORDER,
                ),
                (
                    "LINEBELOW",
                    (0, 0),
                    (-1, 0),
                    0.8,
                    BLUE,
                ),

                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
            ])
        )

        story.append(cat_summary_tb)

        # =========================================================
        # PAGE 5
        # =========================================================

        story.append(PageBreak())

        # ---------------------------------------------------------
        # Numerical plots
        # ---------------------------------------------------------

        story.append(
            Paragraph(
                "Categorical Analysis",
                section_style,
            )
        )

        story.append(
            Paragraph(
                "Distribution, Classes, bar",
                subtitle_style,
            )
        )

        categorical_fig = self.profiler.categorical.plot()

        cat_fig_img = Image("categorical_plots.png")

        cat_fig_img.drawWidth = 400
        cat_fig_img.drawHeight = 650

        story.append(cat_fig_img)

        # ---------------------------------------------------------
        # Build
        # ---------------------------------------------------------
        doc.build(
            story,
            onFirstPage=add_metadata,
            onLaterPages=add_metadata,
        )   

        print(f"Report is saved into `{filepath}`")