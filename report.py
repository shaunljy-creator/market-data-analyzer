# report.py
import io
import logging
from datetime import date

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — required for PDF generation
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, Image, NextPageTemplate,
                                 PageBreak, PageTemplate, Paragraph, Spacer, Table,
                                 TableStyle)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

from config import COL_CLOSE, WINDOW_MA, WINDOW_VOL

# ========================================================
# BLOOMBERG THEME COLOURS

BG       = colors.HexColor("#0A0A0A")
PANEL    = colors.HexColor("#1A1A1A")
ORANGE   = colors.HexColor("#FF6B00")
TEXT     = colors.HexColor("#E0E0E0")
GREY     = colors.HexColor("#2A2A2A")
RED      = colors.HexColor("#FF4444")
GREEN    = colors.HexColor("#44FF88")
WHITE    = colors.white

MPL_THEME = {
    "figure.facecolor":  "#0A0A0A",
    "axes.facecolor":    "#1A1A1A",
    "axes.edgecolor":    "#2A2A2A",
    "axes.labelcolor":   "#E0E0E0",
    "axes.titlecolor":   "#FF6B00",
    "xtick.color":       "#E0E0E0",
    "ytick.color":       "#E0E0E0",
    "text.color":        "#E0E0E0",
    "grid.color":        "#2A2A2A",
    "legend.facecolor":  "#1A1A1A",
    "legend.edgecolor":  "#2A2A2A",
    "font.family":       "monospace",
}

PAGE_W, PAGE_H = A4

# ========================================================
# STYLES

def get_styles():
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "BloombergTitle",
        fontName="Courier-Bold",
        fontSize=24,
        textColor=ORANGE,
        alignment=TA_CENTER,
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "BloombergSubtitle",
        fontName="Courier",
        fontSize=10,
        textColor=TEXT,
        alignment=TA_CENTER,
        spaceAfter=4,
    )
    section_style = ParagraphStyle(
        "BloombergSection",
        fontName="Courier-Bold",
        fontSize=13,
        textColor=ORANGE,
        spaceBefore=8,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "BloombergBody",
        fontName="Courier",
        fontSize=9,
        textColor=TEXT,
        spaceAfter=3,
    )
    return title_style, subtitle_style, section_style, body_style

# ========================================================
# PAGE TEMPLATES (header/footer on every page)

def make_page_templates():
    def header_footer(canvas, doc):
        canvas.saveState()

        # full page background
        canvas.setFillColor(BG)
        canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)

        # top orange bar
        canvas.setFillColor(ORANGE)
        canvas.rect(0, PAGE_H - 18*mm, PAGE_W, 18*mm, fill=1, stroke=0)

        # header text
        canvas.setFont("Courier-Bold", 11)
        canvas.setFillColor(BG)
        canvas.drawString(12*mm, PAGE_H - 11*mm, "PORTFOLIO ANALYZER")
        canvas.drawRightString(PAGE_W - 12*mm, PAGE_H - 11*mm,
                               f"Generated: {date.today().strftime('%d %b %Y')}")

        # bottom bar
        canvas.setFillColor(GREY)
        canvas.rect(0, 0, PAGE_W, 10*mm, fill=1, stroke=0)

        # footer text
        canvas.setFont("Courier", 7)
        canvas.setFillColor(TEXT)
        canvas.drawString(12*mm, 3.5*mm, "FOR INFORMATIONAL PURPOSES ONLY — NOT FINANCIAL ADVICE")
        canvas.drawRightString(PAGE_W - 12*mm, 3.5*mm, f"Page {doc.page}")

        canvas.restoreState()

    frame = Frame(12*mm, 14*mm, PAGE_W - 24*mm, PAGE_H - 36*mm, id="main")
    return [PageTemplate(id="bloomberg", frames=[frame], onPage=header_footer)]

# ========================================================
# TABLE STYLE

def metrics_table_style():
    return TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  ORANGE),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  BG),
        ("FONTNAME",      (0, 0), (-1, 0),  "Courier-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0),  9),
        ("BACKGROUND",    (0, 1), (-1, -1), PANEL),
        ("TEXTCOLOR",     (0, 1), (-1, -1), TEXT),
        ("FONTNAME",      (0, 1), (-1, -1), "Courier"),
        ("FONTSIZE",      (0, 1), (-1, -1), 9),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [PANEL, BG]),
        ("GRID",          (0, 0), (-1, -1), 0.5, GREY),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ])

# ========================================================
# CHART → IMAGE HELPER

def fig_to_image(fig, width_mm=170, height_mm=110) -> Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return Image(buf, width=width_mm*mm, height=height_mm*mm)

# ========================================================
# CHART BUILDERS

def build_ticker_chart(ticker_df: pd.DataFrame, ticker: str,
                        window_ma: int, window_vol: int) -> Image:
    df = ticker_df.reset_index()
    with mpl.rc_context(MPL_THEME):
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f"{ticker} — Technical Analysis", color="#FF6B00",
                     fontsize=13, fontweight="bold")

        # Price + MA + BB
        axes[0].plot(df[COL_CLOSE],    color="#4A9EFF", label="Close",        linewidth=1)
        axes[0].plot(df["MA"],         color="#FF6B00", label=f"MA{window_ma}",linewidth=1, linestyle="--")
        axes[0].plot(df["BB_Upper"],   color="#888888", label="BB",            linewidth=0.8, linestyle="--")
        axes[0].plot(df["BB_Lower"],   color="#888888",                        linewidth=0.8, linestyle="--")
        axes[0].fill_between(range(len(df)), df["BB_Upper"], df["BB_Lower"],
                             alpha=0.1, color="#888888")
        axes[0].set_ylabel("Price")
        axes[0].legend(fontsize=7)
        axes[0].grid(True, alpha=0.3)

        # Cumulative Returns + Drawdown
        axes[1].plot(df["Cumulative Returns"], color="#44FF88", label="Cum. Returns", linewidth=1)
        axes[1].fill_between(range(len(df)), df["Drawdown"], 0,
                             color="#FF4444", alpha=0.3, label="Drawdown")
        axes[1].axhline(0, color="#888888", linewidth=0.6, linestyle="--")
        axes[1].yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
        axes[1].set_ylabel("Return")
        axes[1].legend(fontsize=7)
        axes[1].grid(True, alpha=0.3)

        # Volatility
        axes[2].plot(df["Volatility"], color="#FF4444", label=f"Vol (w={window_vol})", linewidth=1)
        axes[2].set_ylabel("Std Dev")
        axes[2].legend(fontsize=7)
        axes[2].grid(True, alpha=0.3)

        # RSI
        axes[3].plot(df["RSI"], color="#BB88FF", label="RSI", linewidth=1)
        axes[3].axhline(70, color="#FF4444", linewidth=0.7, linestyle="--")
        axes[3].axhline(30, color="#44FF88", linewidth=0.7, linestyle="--")
        axes[3].set_ylim(0, 100)
        axes[3].set_ylabel("RSI")
        axes[3].legend(fontsize=7)
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig_to_image(fig)


def build_heatmap_chart(corr: pd.DataFrame) -> Image:
    with mpl.rc_context(MPL_THEME):
        fig, ax = plt.subplots(figsize=(6, 5))
        cax = ax.imshow(corr.values, cmap="RdBu", vmin=-1, vmax=1)
        fig.colorbar(cax, ax=ax)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.index)))
        ax.set_xticklabels(corr.columns, rotation=45, fontsize=8)
        ax.set_yticklabels(corr.index, fontsize=8)
        for i in range(len(corr)):
            for j in range(len(corr.columns)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                        ha="center", va="center", fontsize=9, color="white")
        ax.set_title("Correlation Matrix")
        plt.tight_layout()
        return fig_to_image(fig, width_mm=120, height_mm=100)


def build_frontier_chart(frontier: pd.DataFrame,
                          port_vol: float, port_return: float) -> Image:
    with mpl.rc_context(MPL_THEME):
        fig, ax = plt.subplots(figsize=(10, 6))
        sc = ax.scatter(frontier["Volatility"], frontier["Return"],
                        c=frontier["Sharpe"], cmap="viridis", s=5, alpha=0.5)
        fig.colorbar(sc, ax=ax, label="Sharpe Ratio")

        max_s = frontier.loc[frontier["Sharpe"].idxmax()]
        min_v = frontier.loc[frontier["Volatility"].idxmin()]

        ax.scatter(max_s["Volatility"], max_s["Return"],
                   color="#FF4444", s=150, marker="*", zorder=5, label="Max Sharpe")
        ax.scatter(min_v["Volatility"], min_v["Return"],
                   color="#4A9EFF", s=150, marker="*", zorder=5, label="Min Volatility")
        ax.scatter(port_vol, port_return,
                   color="#FF6B00", s=150, marker="D", zorder=5, label="Your Portfolio")

        ax.set_xlabel("Annualized Volatility")
        ax.set_ylabel("Annualized Return")
        ax.set_title("Efficient Frontier")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig_to_image(fig, width_mm=160, height_mm=100)

# ========================================================
# MAIN REPORT GENERATOR

def generate_report(
    ticker_data:    dict,           # {ticker: ticker_df}
    ticker_metrics: dict,           # {ticker: {sharpe, beta, alpha, max_dd, var_hist, var_para, cvar}}
    returns_matrix: pd.DataFrame,
    corr_matrix:    pd.DataFrame,
    frontier:       pd.DataFrame,
    port_metrics:   dict,           # {return, vol, sharpe, beta, alpha}
    port_vol:       float,
    port_return:    float,
    window_ma:      int = WINDOW_MA,
    window_vol:     int = WINDOW_VOL,
) -> bytes:

    buf    = io.BytesIO()
    title_style, subtitle_style, section_style, body_style = get_styles()

    doc = BaseDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=12*mm, rightMargin=12*mm,
        topMargin=22*mm,  bottomMargin=14*mm,
    )
    doc.addPageTemplates(make_page_templates())

    story = []

    # ── Cover page ───────────────────────────────────────
    story.append(Spacer(1, 30*mm))
    story.append(Paragraph("PORTFOLIO ANALYSIS REPORT", title_style))
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph(f"Generated: {date.today().strftime('%d %B %Y')}", subtitle_style))
    story.append(Paragraph(f"Tickers: {', '.join(ticker_data.keys())}", subtitle_style))
    story.append(Spacer(1, 10*mm))

    # portfolio summary table on cover
    story.append(Paragraph("Portfolio Summary", section_style))
    port_table_data = [
        ["Metric", "Value"],
        ["Annualized Return",     f"{port_metrics['return']:.4f}"],
        ["Annualized Volatility", f"{port_metrics['vol']:.4f}"],
        ["Sharpe Ratio",          f"{port_metrics['sharpe']:.4f}"],
        ["Beta",                  f"{port_metrics['beta']:.4f}"],
        ["Alpha",                 f"{port_metrics['alpha']:.4f}"],
    ]
    port_table = Table(port_table_data, colWidths=[90*mm, 80*mm])
    port_table.setStyle(metrics_table_style())
    story.append(port_table)
    story.append(PageBreak())

    # ── Per-ticker pages ─────────────────────────────────
    for ticker, df in ticker_data.items():
        m = ticker_metrics[ticker]

        story.append(Paragraph(f"{ticker} — Analysis", section_style))
        story.append(Spacer(1, 3*mm))

        # metrics table
        table_data = [
            ["Metric",           "Value"],
            ["Sharpe Ratio",     f"{m['sharpe']:.4f}"],
            ["Beta",             f"{m['beta']:.4f}"],
            ["Alpha",            f"{m['alpha']:.4f}"],
            ["Max Drawdown",     f"{m['max_dd']:.2%}"],
            ["Historical VaR",   f"{m['var_hist']:.2%}"],
            ["Parametric VaR",   f"{m['var_para']:.2%}"],
            ["CVaR",             f"{m['cvar']:.2%}"],
        ]
        t = Table(table_data, colWidths=[90*mm, 80*mm])
        t.setStyle(metrics_table_style())
        story.append(t)
        story.append(Spacer(1, 4*mm))

        # charts
        story.append(build_ticker_chart(df, ticker, window_ma, window_vol))
        story.append(PageBreak())

    # ── Portfolio page ───────────────────────────────────
    story.append(Paragraph("Portfolio — Correlation & Efficient Frontier", section_style))
    story.append(Spacer(1, 3*mm))
    story.append(build_heatmap_chart(corr_matrix))
    story.append(Spacer(1, 4*mm))
    story.append(build_frontier_chart(frontier, port_vol, port_return))

    doc.build(story)
    buf.seek(0)
    logging.info("PDF report generated successfully")
    return buf.read()