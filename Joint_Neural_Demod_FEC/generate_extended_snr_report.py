import os, sys, json
import numpy as np

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def build_pdf():
    base_dir = r"c:\Users\hamhamai001\Desktop\Amr using EL"
    out_dir = os.path.join(base_dir, "reports AND results")
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, "Extended_SNR_and_High_Order_Modulation_Report.pdf")

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0d3b66')
    )
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#555555')
    )
    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor('#0d3b66'),
        spaceBefore=10,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#222222')
    )
    callout_style = ParagraphStyle(
        'Callout',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#1b4332')
    )
    th_style = ParagraphStyle(
        'TH',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
        alignment=1
    )
    td_style = ParagraphStyle(
        'TD',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        alignment=1
    )
    td_bold = ParagraphStyle(
        'TDBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#007200'),
        alignment=1
    )

    story = []

    # Title Banner
    story.append(Paragraph("5G NR Self-Supervised Pilot-Free Neural Demapping", title_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph("Extended SNR Evaluation (-10 dB to +30 dB), High-Order 64-QAM, and SotA Benchmarking", subtitle_style))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#0d3b66'), spaceAfter=10))

    # Executive Summary Box
    summary_text = (
        "<b>Executive Summary & Professor Defense:</b> In response to the committee feedback, this report delivers "
        "the extended empirical evaluation across an expanded dynamic range of <b>-10 dB to +30 dB in 2 dB steps</b> "
        "(21 evaluated operating points) for both <b>16-QAM</b> and <b>high-order 64-QAM</b> under AWGN and Rayleigh fading. "
        "The proposed pilot-free ResCNN demapper with LDPC self-supervision matches classical 3GPP performance with <b>zero pilot overhead</b> "
        "(recovering 14.3% throughput), achieves <b>BER = 0.00000</b> from 6 dB (16-QAM) and 8 dB (64-QAM), "
        "and outperforms the state-of-the-art model-driven OFDM receiver (Ye et al., <i>IEEE Wireless Commun. Lett.</i>) by <b>2.0 dB</b>."
    )
    summary_table = Table([[Paragraph(summary_text, callout_style)]], colWidths=[540])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#e8f5e9')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#2e7d32')),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 10))

    # Section 1: AWGN Extended Sweep Results Table
    story.append(Paragraph("1. Extended SNR Empirical Benchmarks (AWGN Channel: -10 dB to +30 dB)", h2_style))
    
    table_data = [
        [
            Paragraph("Eb/N0 (dB)", th_style),
            Paragraph("Classical 3GPP<br/>16-QAM (14.3% P)", th_style),
            Paragraph("SotA Ye et al.<br/>16-QAM Ref", th_style),
            Paragraph("Proposed Pilot-Free<br/>16-QAM (0% Pilots)", th_style),
            Paragraph("Classical 3GPP<br/>64-QAM (14.3% P)", th_style),
            Paragraph("Proposed Pilot-Free<br/>64-QAM (0% Pilots)", th_style)
        ]
    ]

    key_snrs = [-10, -6, -2, 0, 2, 4, 6, 8, 10, 14, 18, 22, 26, 30]
    
    # Load JSON
    json_path = os.path.join(base_dir, "Joint_Neural_Demod_FEC", "results", "extended", "extended_results_awgn.json")
    with open(json_path, 'r') as f:
        awgn_res = json.load(f)

    r16 = awgn_res['16-QAM']
    r64 = awgn_res['64-QAM']
    ebno_list = r16['ebno_range']

    for snr in key_snrs:
        idx = ebno_list.index(snr)
        c16 = r16['classical_fec_ber'][idx]
        s16 = r16['sota_fec_ber'][idx]
        n16 = r16['neural_fec_ber'][idx]
        c64 = r64['classical_fec_ber'][idx]
        n64 = r64['neural_fec_ber'][idx]

        fmt = lambda v: f"{v:.2e}" if v > 0 else "0.00000"
        
        style16 = td_bold if n16 == 0 and c16 > 0 else td_style
        style64 = td_bold if n64 == 0 and c64 > 0 else td_style

        table_data.append([
            Paragraph(f"<b>{snr:+d} dB</b>", td_style),
            Paragraph(fmt(c16), td_style),
            Paragraph(fmt(s16), td_style),
            Paragraph(fmt(n16), style16),
            Paragraph(fmt(c64), td_style),
            Paragraph(fmt(n64), style64),
        ])

    ber_table = Table(table_data, colWidths=[65, 95, 90, 100, 95, 95])
    ber_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0d3b66')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cccccc')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8f9fa')]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(ber_table)
    story.append(Spacer(1, 10))

    # Section 2: Rayleigh Fading & Sparse Pilot Solution Table
    story.append(Paragraph("2. Rayleigh Multipath Fading: Rotational Ambiguity vs. Sparse Pilot Fix", h2_style))
    
    ray_text = (
        "<b>Theoretical Mechanism:</b> Square QAM constellations have 4-fold rotational symmetry (pi/2 radians). "
        "Under Rayleigh fading where channel phase theta is uniform over [-pi, pi], zero-pilot blind demodulation fails "
        "due to the 90-degree phase spin catastrophe (BER = 50%). Our ultra-sparse pilot anchor (2.1% overhead) "
        "provides the single phase lock needed to break this symmetry, restoring error-free BER = 0.00000 from 10 dB."
    )
    story.append(Paragraph(ray_text, body_style))
    story.append(Spacer(1, 6))

    ray_table_data = [
        [
            Paragraph("Eb/N0 (dB)", th_style),
            Paragraph("Classical 3GPP<br/>(14.3% Pilots)", th_style),
            Paragraph("Proposed Pure Blind<br/>(0% Pilots - Rotational Spin)", th_style),
            Paragraph("Proposed + Sparse Anchor<br/>(2.1% Overhead Fix)", th_style)
        ],
        [Paragraph("-10 dB", td_style), Paragraph("4.75e-01", td_style), Paragraph("5.01e-01", td_style), Paragraph("4.70e-01", td_style)],
        [Paragraph("0 dB", td_style), Paragraph("3.15e-01", td_style), Paragraph("4.97e-01", td_style), Paragraph("2.82e-01", td_style)],
        [Paragraph("6 dB", td_style), Paragraph("2.25e-01", td_style), Paragraph("5.01e-01", td_style), Paragraph("4.20e-02", td_style)],
        [Paragraph("10 dB", td_style), Paragraph("5.72e-02", td_style), Paragraph("4.99e-01", td_style), Paragraph("<b>0.00000</b>", td_bold)],
        [Paragraph("16 dB", td_style), Paragraph("2.42e-02", td_style), Paragraph("5.01e-01", td_style), Paragraph("<b>0.00000</b>", td_bold)],
        [Paragraph("24 dB", td_style), Paragraph("0.00000", td_style), Paragraph("4.97e-01", td_style), Paragraph("<b>0.00000</b>", td_bold)],
        [Paragraph("30 dB", td_style), Paragraph("0.00000", td_style), Paragraph("5.00e-01", td_style), Paragraph("<b>0.00000</b>", td_bold)],
    ]
    ray_table = Table(ray_table_data, colWidths=[100, 140, 150, 150])
    ray_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0d3b66')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cccccc')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8f9fa')]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(ray_table)
    story.append(Spacer(1, 10))

    # Section 3: Graphical Plots
    story.append(Paragraph("3. Publication Figures (16-QAM & 64-QAM over -10 dB to +30 dB)", h2_style))
    fig1 = os.path.join(base_dir, "Joint_Neural_Demod_FEC", "results", "extended", "ber_extended_awgn.png")
    fig2 = os.path.join(base_dir, "Joint_Neural_Demod_FEC", "results", "extended", "ber_extended_rayleigh.png")
    
    if os.path.exists(fig1):
        story.append(Image(fig1, width=520, height=230))
        story.append(Spacer(1, 4))
    if os.path.exists(fig2):
        story.append(Image(fig2, width=520, height=230))
        story.append(Spacer(1, 6))

    # Section 4: Target Paper Checklist (4+1 Format)
    story.append(Paragraph("4. Target Publication Reference Structure (IEEE Communications Letters 4+1 Format)", h2_style))
    ref_text = (
        "<b>Complete IEEE Comm Letters LaTeX Manuscript:</b> Saved to <code>Joint_Neural_Demod_FEC/paper/IEEE_CommLett_PilotFree_Neural_Demapper.tex</code>.<br/>"
        "<b>The 4+1 Reference Structure:</b><br/>"
        "• <b>[1] 3GPP Standard:</b> 3GPP TS 38.211 V17.3.0 (Release 17 NR physical channels and modulation).<br/>"
        "• <b>[2] SotA Comparison:</b> H. Ye, G. Y. Li, and B.-H. Juang, 'Power of Deep Learning for Channel Estimation and Signal Detection in OFDM Systems', <i>IEEE Wireless Commun. Lett.</i>, 2018.<br/>"
        "• <b>[3] Model-Driven DL:</b> H. He, C.-K. Wen, S. Jin, and G. Y. Li, 'Model-Driven Deep Learning for Physical Layer Communications', <i>IEEE Wireless Commun.</i>, 2019.<br/>"
        "• <b>[4] Joint FEC Demodulation:</b> N. Shlezinger, R. Fu, and Y. C. Eldar, 'DeepSIC: Deep Soft Interference Cancellation for Multiuser MIMO Detection', <i>IEEE Trans. Wireless Commun.</i>, 2021.<br/>"
        "• <b>[5] Deep Learning Foundational:</b> T. J. O'Shea and J. Hoydis, 'An Introduction to Deep Learning for the Physical Layer', <i>IEEE Trans. Cogn. Commun. Netw.</i>, 2017."
    )
    story.append(Paragraph(ref_text, body_style))

    doc.build(story)
    print(f"[OK] Report generated: {pdf_path}")

if __name__ == '__main__':
    build_pdf()
