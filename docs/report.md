# Report

`Report` generates a PDF report from a `Profiler` instance.

It is normally accessed as:

```python
from lochan_eda import Profiler

profile = Profiler(df)
profile.report.save("report.pdf")
```

## `save(filepath="report.pdf")`

```python
profile.report.save("eda_report.pdf")
```

### Parameter

| Parameter | Type | Default | Description |
|---|---|---|---|
| `filepath` | `str` | `report.pdf` | Path where the PDF report is written. |

The generated report contains the dataset overview and the report sections implemented by the current `Report` class.

The PDF is built with ReportLab and includes a consistent header, footer, page numbering, and tabular presentation of overview information.
