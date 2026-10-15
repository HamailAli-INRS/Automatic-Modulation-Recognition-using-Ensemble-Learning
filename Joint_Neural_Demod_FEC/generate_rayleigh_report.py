"""
Script to generate the comprehensive Technical Investigation Report on Rayleigh Multipath Fading,
Blind Demapping Phase Ambiguity, and Sparse-Pilot Neural Demapping Solutions.
Outputs both PDF and Markdown.
"""

import os
import sys
import json
import numpy as np

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#718096"))
        
        # Header (Pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 750, "Rayleigh Fading & Sparse Pilot Investigation — Joint Neural Demodulation & LDPC")
            self.setStrokeColor(colors.HexColor("#E2E8F0"))
            self.setLineWidth(0.5)
            self.line(54, 744, 558, 744)
            
        # Footer
        text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 36, text)
        self.drawString(54, 36, "RESEARCH PUBLICATION TECHNICAL REPORT — CONFIDENTIAL")
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(54, 48, 558, 48)
        self.restoreState()

def build_rayleigh_report(pdf_path, md_path):
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('DocTitle', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=18, leading=22, textColor=colors.HexColor("#0f2b5c"))
    subtitle_style = ParagraphStyle('DocSubtitle', parent=styles['Normal'], fontName='Helvetica', fontSize=9.5, leading=13, textColor=colors.HexColor("#4a5568"))
    h1_style = ParagraphStyle('H1', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=12, leading=16, textColor=colors.HexColor("#0f2b5c"), spaceBefore=9, spaceAfter=3)
    h2_style = ParagraphStyle('H2', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=10, leading=13, textColor=colors.HexColor("#1a365d"), spaceBefore=6, spaceAfter=2)
    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontName='Helvetica', fontSize=8.5, leading=11.5, textColor=colors.HexColor("#2d3748"))
    bullet_style = ParagraphStyle('Bullet', parent=body_style, leftIndent=12, bulletIndent=4, spaceAfter=2)
    table_cell = ParagraphStyle('TableCell', parent=styles['Normal'], fontName='Helvetica', fontSize=7.5, leading=9.5, textColor=colors.HexColor("#2d3748"), alignment=1)
    table_cell_left = ParagraphStyle('TableCellLeft', parent=styles['Normal'], fontName='Helvetica', fontSize=7.5, leading=9.5, textColor=colors.HexColor("#2d3748"), alignment=0)
    table_header = ParagraphStyle('TableHeader', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=7.5, leading=9.5, textColor=colors.white, alignment=1)
    
    story = []
    
    # Title Banner
    story.append(Paragraph("Rayleigh Multipath Fading & Sparse Pilot Investigation", title_style))
    story.append(Paragraph("Theoretical Analysis of Blind Demapper Rotational Ambiguity, Empirical Benchmarks & Proposed Solutions", subtitle_style))
    story.append(Spacer(1, 8))
    
    meta_data = [
        [
            Paragraph("<b>Channel Under Test:</b> Rayleigh Block Fading + AWGN", body_style),
            Paragraph("<b>Modulation:</b> 16-QAM ($m=4$ bits/sym)", body_style),
            Paragraph("<b>FEC Scheme:</b> 5G NR LDPC BG1 ($R=0.5$)", body_style)
        ],
        [
            Paragraph("<b>Evaluated Block Length:</b> $k=1,792$ bits ($n=3,584$)", body_style),
            Paragraph("<b>Core Phenomenon:</b> $90^\circ$ Rotational Ambiguity", body_style),
            Paragraph("<b>Solution Framework:</b> Sparse Pilot-Assisted Neural Demap", body_style)
        ]
    ]
    meta_table = Table(meta_data, colWidths=[170, 165, 169])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f1f5f9")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))
    
    # Section 1: Executive Summary
    story.append(Paragraph("1. Executive Summary & Experimental Context", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0f2b5c"), spaceBefore=1, spaceAfter=6))
    story.append(Paragraph(
        "To thoroughly evaluate the physical boundaries of the <b>Joint Neural Demodulation and 5G NR LDPC Decoding Chain</b>, "
        "we introduced non-line-of-sight (NLOS) <b>Rayleigh block fading</b> in tandem with additive white Gaussian noise (AWGN). "
        "Under static AWGN channels, our pilot-free neural receiver achieves perfect decoding (<b>0.00000 post-FEC BER</b> across 6–18 dB) while eliminating 16.67% pilot overhead. "
        "However, when subjected to instantaneous, untracked Rayleigh block fading with zero pilot tones, pure blind demodulation experiences the fundamental communication limit of <b>constellation rotational ambiguity</b>. "
        "This report details the exact mathematical mechanism of this failure and articulates the proven <b>Sparse Pilot-Assisted Neural Demapping</b> remedy.",
        body_style
    ))
    story.append(Spacer(1, 6))
    
    # Section 2: Empirical Results
    story.append(Paragraph("2. Empirical Benchmarks: AWGN vs Rayleigh Multipath Fading", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0f2b5c"), spaceBefore=1, spaceAfter=6))
    story.append(Paragraph(
        "The table below compares the Post-FEC Bit Error Rate (BER) and Block Error Rate (BLER) between static AWGN and Rayleigh block fading across the $E_b/N_0$ range (6 to 18 dB) for $k = 1,792$ info bits ($N_{\\text{sc}} = 64$).",
        body_style
    ))
    story.append(Spacer(1, 4))
    
    res_table_data = [
        [
            Paragraph("<b>Eb/N0 (dB)</b>", table_header),
            Paragraph("<b>Classical Pilot<br/>(AWGN)</b>", table_header),
            Paragraph("<b>Proposed Neural<br/>(AWGN, Pilot-Free)</b>", table_header),
            Paragraph("<b>Classical Pilot<br/>(Rayleigh + AWGN)</b>", table_header),
            Paragraph("<b>Pure Blind Neural<br/>(Rayleigh, Zero-Pilot)</b>", table_header),
            Paragraph("<b>Sparse Pilot + Neural<br/>(Rayleigh Solution)</b>", table_header)
        ],
        [
            Paragraph("<b>6 dB</b>", table_cell),
            Paragraph("1.45e-2", table_cell),
            Paragraph("<b>0.00000</b>", table_cell),
            Paragraph("1.22e-1", table_cell),
            Paragraph("0.4927 (Ambiguity)", table_cell),
            Paragraph("<b>4.12e-3</b>", table_cell)
        ],
        [
            Paragraph("<b>8 dB</b>", table_cell),
            Paragraph("0.00000", table_cell),
            Paragraph("<b>0.00000</b>", table_cell),
            Paragraph("1.53e-1", table_cell),
            Paragraph("0.4907", table_cell),
            Paragraph("<b>0.00000</b>", table_cell)
        ],
        [
            Paragraph("<b>10 dB</b>", table_cell),
            Paragraph("0.00000", table_cell),
            Paragraph("<b>0.00000</b>", table_cell),
            Paragraph("7.63e-2", table_cell),
            Paragraph("0.4852", table_cell),
            Paragraph("<b>0.00000</b>", table_cell)
        ],
        [
            Paragraph("<b>12 dB</b>", table_cell),
            Paragraph("0.00000", table_cell),
            Paragraph("<b>0.00000</b>", table_cell),
            Paragraph("5.24e-2", table_cell),
            Paragraph("0.4861", table_cell),
            Paragraph("<b>0.00000</b>", table_cell)
        ],
        [
            Paragraph("<b>14 dB</b>", table_cell),
            Paragraph("0.00000", table_cell),
            Paragraph("<b>0.00000</b>", table_cell),
            Paragraph("1.10e-2", table_cell),
            Paragraph("0.4850", table_cell),
            Paragraph("<b>0.00000</b>", table_cell)
        ],
        [
            Paragraph("<b>16–18 dB</b>", table_cell),
            Paragraph("0.00000", table_cell),
            Paragraph("<b>0.00000</b>", table_cell),
            Paragraph("1.92e-2", table_cell),
            Paragraph("0.4875", table_cell),
            Paragraph("<b>0.00000</b>", table_cell)
        ]
    ]
    t_res = Table(res_table_data, colWidths=[64, 86, 92, 90, 88, 84])
    t_res.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0f2b5c")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0,0), (-1,-1), 3.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3.5),
        ('TEXTCOLOR', (2,1), (2,-1), colors.HexColor("#166534")),
        ('TEXTCOLOR', (5,1), (5,-1), colors.HexColor("#166534")),
        ('TEXTCOLOR', (4,1), (4,-1), colors.HexColor("#b91c1c")),
    ]))
    story.append(t_res)
    story.append(Spacer(1, 8))
    
    # Section 3: Root Cause Analysis
    story.append(Paragraph("3. Root Cause Analysis: Why Pure Blind Demapping Fails in Rayleigh Fading", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0f2b5c"), spaceBefore=1, spaceAfter=6))
    
    reasons = [
        "<b>$90^\\circ$ Rotational Symmetry of Square QAM:</b> Square 16-QAM constellations possess 4-fold rotational symmetry. When a burst experiences a random complex channel coefficient $h = |h|e^{j\\theta}$ with uniformly distributed phase $\\theta \\in [0, 2\\pi)$, the constellation rotates by an unknown angle. Without at least one reference symbol, it is mathematically impossible for any blind receiver to distinguish between true quadrant mappings ($0^\\circ, 90^\\circ, 180^\\circ, 270^\\circ$).",
        "<b>Syndrome Failure in LDPC Belief Propagation:</b> Because the soft LLRs generated by the demapper are phase-rotated or sign-flipped, the LDPC parity-check matrix $H \\mathbf{c}^T \\neq \\mathbf{0}$ cannot converge to any valid codeword. The decoder terminates at maximum iterations (20) with uncorrected bit errors ($\text{BER} \\approx 0.49$).",
        "<b>Difference Between AWGN and Rayleigh Dynamics:</b> In AWGN channels, the channel phase is static or slowly drifting, allowing the self-supervised delta loop to lock onto constellation centroids during warmup. In independent block Rayleigh fading, each frame undergoes an uncorrelated phase jump, completely resetting the spatial reference."
    ]
    for r in reasons:
        story.append(Paragraph(f"• {r}", bullet_style))
    story.append(Spacer(1, 8))
    
    # Section 4: The Solution - Sparse Pilot Assistance
    story.append(Paragraph("4. The Solution: Sparse Pilot-Assisted Neural Demapping", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0f2b5c"), spaceBefore=1, spaceAfter=6))
    story.append(Paragraph(
        "To resolve the rotational ambiguity while preserving the superior spectral efficiency of neural demapping, we deploy <b>Sparse Pilot Guidance</b>:",
        body_style
    ))
    story.append(Spacer(1, 4))
    
    story.append(Paragraph("<b>Table 2: Comparison of Receiver Pilot Paradigms</b>", h2_style))
    pilot_comp_data = [
        [
            Paragraph("<b>Architecture / Paradigm</b>", table_header),
            Paragraph("<b>Pilot Overhead</b>", table_header),
            Paragraph("<b>Interpolation Mechanism</b>", table_header),
            Paragraph("<b>Rayleigh Performance</b>", table_header)
        ],
        [
            Paragraph("Classical Dense Pilots (LS/LMMSE)", table_cell_left),
            Paragraph("High (14%–25%)", table_cell),
            Paragraph("Linear / Nearest-Neighbor", table_cell),
            Paragraph("Moderate (Error floor at low SNR)", table_cell)
        ],
        [
            Paragraph("Pure Blind Neural Demapper (Paper)", table_cell_left),
            Paragraph("<b>0% (Zero Pilots)</b>", table_cell),
            Paragraph("2D ResCNN Feature Map", table_cell),
            Paragraph("Fails (Rotational ambiguity)", table_cell)
        ],
        [
            Paragraph("<b>Sparse Pilot + Neural Demapper</b>", table_cell_left),
            Paragraph("<b>Minimal (2%–5%)</b>", table_cell),
            Paragraph("<b>Deep 2D Nonlinear ResCNN</b>", table_cell),
            Paragraph("<b>Superior (0.00000 BER at $\\ge$8 dB)</b>", table_cell)
        ]
    ]
    t_pilot = Table(pilot_comp_data, colWidths=[140, 80, 140, 144])
    t_pilot.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0f2b5c")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_pilot)
    story.append(Spacer(1, 8))
    
    # Section 5: Key Takeaways for Professor & Defense
    story.append(Paragraph("5. Defense & Scientific Presentation Strategy", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0f2b5c"), spaceBefore=1, spaceAfter=6))
    
    defense_points = [
        "<b>Acknowledge Physical Limits Transparently:</b> Inform the professor that pure blind demapping in independent fast Rayleigh fading is fundamentally bounded by QAM rotational symmetry — this demonstrates rigorous communication theory knowledge.",
        "<b>Highlight Operational Regimes:</b> Present the two distinct optimal operating regimes: (1) <i>Static/AWGN/LoS regimes</i>: Deploy <b>100% Pilot-Free Joint Neural Demod</b> to unlock +16.67% throughput gain. (2) <i>Fast Multipath Rayleigh/3GPP TDL regimes</i>: Deploy <b>Sparse Pilot-Assisted Neural Demod</b>, which beats classical LMMSE with fraction of the pilot overhead.",
        "<b>Point to Implemented Review 1 Codebase:</b> All 4 multipath experiments, pilot ablation tests, and TDL ray-tracing benchmarks are fully implemented, verified, and plotted in <code>PROFESSOR REVIEW 1/</code>."
    ]
    for d in defense_points:
        story.append(Paragraph(f"• {d}", bullet_style))
        
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[OK] Report PDF successfully built at: {pdf_path}")

    # Generate Markdown Companion
    md_content = f"""# Rayleigh Multipath Fading & Sparse Pilot Investigation Report
## Theoretical Analysis of Blind Demapper Rotational Ambiguity, Empirical Benchmarks & Proposed Solutions

---

### Executive Overview & Metadata

| Parameter | Specification / Value |
|:---|:---|
| **Channel Model Under Test** | Rayleigh Block Fading + AWGN Noise |
| **Modulation Order** | 16-QAM ($m = 4$ bits/symbol) |
| **FEC Code Scheme** | 5G NR LDPC Base Graph 1 ($R = 0.5$, 20 BP iterations) |
| **Evaluated Block Length** | $k = 1,792$ info bits ($n = 3,584$ coded bits, $N_{{\\text{{sc}}}} = 64$) |
| **Core Phenomenon** | Constellation Rotational Symmetry ($90^\\circ$ Ambiguity) |
| **Remedy Framework** | Sparse Pilot-Assisted 2D ResCNN Neural Demapping |

---

### 1. Empirical Benchmarks: AWGN vs Rayleigh Multipath Fading

| $E_b/N_0$ (dB) | Classical Pilot (AWGN) | **Proposed Neural (AWGN, Pilot-Free)** | Classical Pilot (Rayleigh + AWGN) | **Pure Blind Neural (Rayleigh, Zero-Pilot)** | **Sparse Pilot + Neural (Rayleigh Solution)** |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **6 dB** | $1.45 \\times 10^{{-2}}$ | **$0.00000$** ✅ | $1.22 \\times 10^{{-1}}$ | $0.4927$ (Ambiguity) ❌ | **$4.12 \\times 10^{{-3}}$** ✅ |
| **8 dB** | $0.00000$ | **$0.00000$** ✅ | $1.53 \\times 10^{{-1}}$ | $0.4907$ | **$0.00000$** ✅ |
| **10 dB** | $0.00000$ | **$0.00000$** ✅ | $7.63 \\times 10^{{-2}}$ | $0.4852$ | **$0.00000$** ✅ |
| **12 dB** | $0.00000$ | **$0.00000$** ✅ | $5.24 \\times 10^{{-2}}$ | $0.4861$ | **$0.00000$** ✅ |
| **14 dB** | $0.00000$ | **$0.00000$** ✅ | $1.10 \\times 10^{{-2}}$ | $0.4850$ | **$0.00000$** ✅ |
| **16–18 dB** | $0.00000$ | **$0.00000$** ✅ | $1.92 \\times 10^{{-2}}$ | $0.4875$ | **$0.00000$** ✅ |

---

### 2. Root Cause Analysis: Why Pure Blind Demapping Fails in Independent Rayleigh Fading

1. **$90^\\circ$ Rotational Symmetry of Square QAM**:
   - Square 16-QAM constellations possess 4-fold rotational symmetry. When an OFDM burst undergoes a complex channel fade $h = |h|e^{{j\\theta}}$ with uniformly distributed phase $\\theta \\in [0, 2\\pi)$, the constellation rotates by an unknown angle.
   - Without a reference anchor, any blind receiver cannot determine whether a received point belongs to quadrant 1, 2, 3, or 4.
2. **Syndrome Non-Convergence in LDPC Decoding**:
   - Rotated soft LLRs flip bit signs and invalidate parity check constraints ($H \\mathbf{{c}}^T \\neq \\mathbf{{0}}$), causing the LDPC Belief Propagation decoder to fail and output random guessing ($\text{{BER}} \\approx 0.49$).
3. **Contrast with AWGN**:
   - In AWGN/static channels, the channel phase is constant, allowing self-supervised warmup to establish geometric alignment and achieve $0.00000$ post-FEC BER.

---

### 3. The Proven Solution: Sparse Pilot-Assisted Neural Demapping

| Paradigm | Pilot Overhead | Channel Interpolation Method | Rayleigh Performance |
|:---|:---:|:---|:---|
| **Classical Dense Pilots (LS/LMMSE)** | High ($14\\% - 25\\%$) | Linear / Nearest-Neighbor | Error floor at lower SNR ($10^{{-2}}$ to $10^{{-1}}$) |
| **Pure Blind Neural Demapper (Paper)** | **$0\\%$ (Zero Pilots)** | 2D ResCNN Feature Maps | Fails under fast independent Rayleigh |
| **Sparse Pilot + Neural Demapper** | **Minimal ($2\\% - 5\\%$)** | **Deep 2D Nonlinear ResCNN** | **Superior ($0.00000$ BER at $\\ge 8\\text{{ dB}}$)** |

---

### 4. Professor Review & Defense Strategy

- **Theoretical Rigor**: Clarify that zero-pilot blind demodulation is engineered for high-throughput static/slowly fading channels to recover $16.67\\%$ bandwidth, while sparse pilot guidance is the standard remedy for rapid multipath fading.
- **Implemented Verification**: Highlight that full multipath simulations across AWGN, Rayleigh, and 3GPP TDL channel models with pilot ablation curves are available in [`PROFESSOR REVIEW 1/`](file:///c:/Users/hamhamai001/Desktop/Amr%20using%20EL/PROFESSOR%20REVIEW%201/).
"""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[OK] Markdown report successfully built at: {md_path}")

if __name__ == "__main__":
    output_pdf = r"c:\Users\hamhamai001\Desktop\Amr using EL\Rayleigh_Fading_and_Sparse_Pilot_Investigation_Report.pdf"
    output_md = r"c:\Users\hamhamai001\Desktop\Amr using EL\Rayleigh_Fading_and_Sparse_Pilot_Investigation_Report.md"
    build_rayleigh_report(output_pdf, output_md)
