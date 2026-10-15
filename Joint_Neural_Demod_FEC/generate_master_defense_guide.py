"""
Script to generate the printable Master Project Understanding & Defense Guide:
- Master_Project_Understanding_and_Defense_Guide.pdf
- Master_Project_Understanding_and_Defense_Guide.md

Designed specifically for printout with clean typography, clear sectioning,
and complete end-to-end coverage of theory, pipeline, equations, and defense Q&A.
"""

import os
import sys

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
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
        self.setFillColor(colors.HexColor("#64748b"))
        
        # Header (Pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 752, "5G NR Joint Neural Demapper & LDPC: Master Understanding & Oral Defense Guide")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(54, 746, 558, 746)
            
        # Footer
        text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 36, text)
        self.drawString(54, 36, "CONFIDENTIAL STUDY GUIDE — PREPARED FOR PROFESSOR MEETING & IEEE DEFENSE")
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(54, 48, 558, 48)
        self.restoreState()

def get_print_styles():
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('DocTitle', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=18, leading=22, textColor=colors.HexColor("#0f2b5c"), alignment=0)
    subtitle_style = ParagraphStyle('DocSubtitle', parent=styles['Normal'], fontName='Helvetica', fontSize=9.5, leading=14, textColor=colors.HexColor("#475569"), spaceBefore=4, spaceAfter=8)
    h1_style = ParagraphStyle('H1', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=12, leading=16, textColor=colors.HexColor("#0f2b5c"), spaceBefore=10, spaceAfter=4, keepWithNext=True)
    h2_style = ParagraphStyle('H2', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=10, leading=13, textColor=colors.HexColor("#1e3a8a"), spaceBefore=7, spaceAfter=3, keepWithNext=True)
    h3_style = ParagraphStyle('H3', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=8.5, leading=11, textColor=colors.HexColor("#0369a1"), spaceBefore=5, spaceAfter=2, keepWithNext=True)
    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontName='Helvetica', fontSize=8.5, leading=12, textColor=colors.HexColor("#1e293b"), spaceAfter=4)
    body_bold = ParagraphStyle('BodyBold', parent=body_style, fontName='Helvetica-Bold')
    callout_style = ParagraphStyle('Callout', parent=styles['Normal'], fontName='Helvetica-Oblique', fontSize=8.5, leading=12, textColor=colors.HexColor("#0f172a"))
    bullet_style = ParagraphStyle('Bullet', parent=body_style, leftIndent=12, bulletIndent=4, spaceAfter=3)
    table_cell = ParagraphStyle('TableCell', parent=styles['Normal'], fontName='Helvetica', fontSize=7.5, leading=10, textColor=colors.HexColor("#1e293b"), alignment=1)
    table_cell_left = ParagraphStyle('TableCellLeft', parent=styles['Normal'], fontName='Helvetica', fontSize=7.5, leading=10, textColor=colors.HexColor("#1e293b"), alignment=0)
    table_header = ParagraphStyle('TableHeader', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=7.5, leading=10, textColor=colors.white, alignment=1)
    
    return {
        'title': title_style, 'subtitle': subtitle_style, 'h1': h1_style, 'h2': h2_style, 'h3': h3_style,
        'body': body_style, 'body_bold': body_bold, 'callout': callout_style, 'bullet': bullet_style,
        'cell': table_cell, 'cell_left': table_cell_left, 'header': table_header
    }

def make_callout(text, st, bg_color="#f8fafc", border_color="#cbd5e1"):
    t = Table([[Paragraph(text, st['body'])]], colWidths=[504])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor(bg_color)),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor(border_color)),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    return t

def build_defense_guide(pdf_path, md_path):
    doc = SimpleDocTemplate(
        pdf_path, pagesize=letter,
        leftMargin=54, rightMargin=54, topMargin=54, bottomMargin=54
    )
    st = get_print_styles()
    story = []

    # ─── TITLE & HEADER ───
    story.append(Paragraph("Master Project Understanding & Oral Defense Guide", st['title']))
    story.append(Paragraph("End-to-End Pipeline Walkthrough, Theoretical Foundations, Equation Decoder & Professor Defense Script", st['subtitle']))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0f2b5c"), spaceBefore=2, spaceAfter=8))

    # Meta banner
    meta_table = Table([
        [
            Paragraph("<b>Target Domain:</b> 5G NR / 6G Physical Layer AI", st['cell_left']),
            Paragraph("<b>Modulation & Code:</b> 16-QAM + 5G LDPC BG1 (R=0.5)", st['cell_left']),
            Paragraph("<b>Core Model:</b> 2D ResCNN Demapper (128K params)", st['cell_left'])
        ],
        [
            Paragraph("<b>Key Gain:</b> +16.67% Spectral Efficiency (Zero Pilots)", st['cell_left']),
            Paragraph("<b>Coding Gain:</b> +2 to +3.8 dB over Classical 3GPP", st['cell_left']),
            Paragraph("<b>Adaptation:</b> Self-Supervised Delta-Loop + Replay Buffer", st['cell_left'])
        ]
    ], colWidths=[168, 168, 168])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f1f5f9")),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#94a3b8")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # ─── CHAPTER 1: THE CORE STORY IN 2 MINUTES ───
    story.append(Paragraph("Chapter 1: The Core Story in Plain English (The 2-Minute Elevator Pitch)", st['h1']))
    story.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor("#0f2b5c"), spaceBefore=1, spaceAfter=6))
    
    story.append(Paragraph(
        "<b>1. The Real-World Wireless Problem (The Pilot Tax):</b><br/>"
        "In standardized 5G New Radio (NR), whenever a mobile phone or base station transmits data across radio waves, "
        "it is forced to waste <b>14.29% of its total bandwidth</b> on fixed benchmark signals called <i>Pilots</i> (Demodulation Reference Signals, DMRS). "
        "Pilots carry zero user information—they exist solely so the receiver can estimate how the wireless channel distorted the signal. "
        "Worse yet, in low-signal environments (low SNR), estimating the channel using noisy pilots causes severe errors that corrupt the entire packet.",
        st['body']
    ))
    story.append(Paragraph(
        "<b>2. Our Invention (Pilot-Free Joint Neural Demapping):</b><br/>"
        "We completely eliminate all pilots (100% data transmission), delivering an immediate, permanent <b>+16.67% spectral efficiency gain</b>. "
        "Instead of relying on fragile channel estimation formulas, we deploy a lightweight <b>2D Residual Convolutional Neural Network (2D ResCNN)</b>. "
        "The CNN processes the entire time-frequency resource grid directly like a 2D image, capturing complex spatial noise correlations and outputting soft bit probabilities.",
        st['body']
    ))
    story.append(Paragraph(
        "<b>3. The Self-Supervised Magic (The Teacher-Student Delta Loop):</b><br/>"
        "How can the CNN learn online without any pilots or ground-truth bit labels? "
        "We pair the CNN with a standardized <b>5G NR LDPC Belief Propagation Decoder</b>. "
        "Think of the CNN as an <i>eager student</i> and the LDPC decoder as a <i>strict mathematical spell-checker</i>:<br/>"
        "• The CNN makes an initial guess of the bits (which contains 10%–20% raw errors).<br/>"
        "• The LDPC decoder uses strict parity-check geometry (<b>H</b> &middot; <b>c</b><sup>T</sup> = <b>0</b>) to correct these errors, producing certified mathematical truth.<br/>"
        "• We compute the difference (the <b>Delta signal &Delta;</b>) between what the CNN guessed and what the LDPC decoder proved.<br/>"
        "• The CNN updates its weights using this Delta signal through an Experience Replay Buffer—<b>learning continuously online without true labels!</b>",
        st['body']
    ))
    story.append(Spacer(1, 8))

    # ─── CHAPTER 2: THE END-TO-END PIPELINE WALKTHROUGH ───
    story.append(Paragraph("Chapter 2: The End-to-End Pipeline Step-by-Step", st['h1']))
    story.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor("#0f2b5c"), spaceBefore=1, spaceAfter=6))

    pipeline_stages = [
        "<b>Stage 1: Information Bit Generation (TX):</b> The user payload begins as a binary vector <b>u</b> &in; {0, 1}<sup>k</sup> (e.g., k = 1,792 bits). "
        "When testing scalability, we expand this up to k = 8,400 bits (the maximum 3GPP LDPC block size).",

        "<b>Stage 2: 5G NR LDPC Encoding (TX):</b> The bits are encoded via a standard 3GPP TS 38.212 rate-1/2 LDPC encoder into a codeword <b>c</b> &in; {0, 1}<sup>n</sup> of length n = 3,584 bits. "
        "The codeword satisfies the parity-check equation <b>H</b> &middot; <b>c</b><sup>T</sup> = <b>0</b> (mod 2).",

        "<b>Stage 3: 16-QAM Modulation (TX):</b> The 3,584 bits are grouped into 4-bit nibbles (M = 4 bits/symbol) and mapped to 896 complex modulation symbols <b>s</b> using Gray coding with unit average power.",

        "<b>Stage 4: Zero-Pilot Resource Grid Mapping (TX):</b> A standard 5G slot contains 14 OFDM symbols &times; 64 subcarriers = 896 Resource Elements (REs). "
        "In our zero-pilot design, <b>all 896 elements carry data symbols</b> (0% pilots). "
        "An IFFT of size 64 with a 16-sample Cyclic Prefix (CP) modulates the grid into the time-domain.",

        "<b>Stage 5: Wireless Channel Dynamics:</b> The signal travels over the wireless channel, experiencing multipath delay spread and thermal noise: Y[l, k] = H[l, k]X[l, k] + W[l, k]. "
        "We evaluate both static AWGN (H = 1) and dispersive Rayleigh multipath block fading.",

        "<b>Stage 6: Receiver Demodulation & FFT (RX):</b> The receiver removes the CP and executes a 64-point FFT, reconstructing the received distorted complex grid Y of shape [14, 64]. "
        "The complex grid is formatted into a 2-channel real tensor [B, 2, 14, 64] (In-Phase and Quadrature).",

        "<b>Stage 7: 2D ResCNN Demapping (RX):</b> The 2-channel grid passes through our 2D ResCNN demapper (3 residual blocks, 48 hidden channels). "
        "It outputs 4 channels corresponding to the 4 bit planes of 16-QAM, scaled by a learnable temperature &alpha; to produce calibrated soft Log-Likelihood Ratios (LLRs) &lambda;.",

        "<b>Stage 8: 5G NR LDPC Belief Propagation Decoding (RX):</b> The soft LLRs enter the LDPC Belief Propagation decoder (20 message-passing iterations), which corrects raw demapper errors and outputs the final decoded user bits <b>u</b>&#770;.",

        "<b>Stage 9: The Self-Supervised Delta Adaptation Loop:</b> The decoded bits are re-encoded: <b>c</b>&#770; = LDPC-Encode(<b>u</b>&#770;). "
        "If raw disagreement d &le; 30%, the pair is pushed into a circular Experience Replay Buffer (capacity 128). "
        "Mini-batches are sampled from the buffer to update the CNN via AdamW with BCEWithLogits loss."
    ]

    for p in pipeline_stages:
        story.append(Paragraph(f"• {p}", st['bullet']))
    story.append(Spacer(1, 8))

    # ─── CHAPTER 3: DICTIONARY OF KEY JARGONS & TECHNOLOGIES ───
    story.append(Paragraph("Chapter 3: Dictionary of Jargons & Technologies", st['h1']))
    story.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor("#0f2b5c"), spaceBefore=1, spaceAfter=6))

    jargons_data = [
        [Paragraph("<b>Term / Technology</b>", st['header']), Paragraph("<b>Plain-English Definition & Why It Matters</b>", st['header']), Paragraph("<b>Implementation in Our Code</b>", st['header'])],
        [
            Paragraph("<b>OFDM Resource Grid</b>", st['cell_left']),
            Paragraph("A 2D matrix of radio frequencies (subcarriers) along the horizontal axis and time slices (OFDM symbols) along the vertical axis. 1 slot = 14 symbols.", st['cell_left']),
            Paragraph("<code>ResourceGrid(num_ofdm_symbols=14, fft_size=64)</code> in Sionna.", st['cell_left'])
        ],
        [
            Paragraph("<b>Demodulation Reference Signals (DMRS)</b>", st['cell_left']),
            Paragraph("Known pilot symbols embedded in the grid to measure channel fading. Wastes 14.29% of bandwidth. Completely eliminated in our scheme.", st['cell_left']),
            Paragraph("Classical uses <code>pilot_pattern='kronecker'</code>; Proposed sets <code>pilot_pattern=None</code>.", st['cell_left'])
        ],
        [
            Paragraph("<b>Log-Likelihood Ratio (LLR)</b>", st['cell_left']),
            Paragraph("Soft confidence score for a bit. LLR &gt; 0 means bit=1; LLR &lt; 0 means bit=0. Magnitude reflects certainty. Essential for LDPC decoding.", st['cell_left']),
            Paragraph("Output logits of 2D ResCNN scaled by learnable parameter <code>self.scale</code>.", st['cell_left'])
        ],
        [
            Paragraph("<b>5G NR LDPC Base Graph 1 (BG1)</b>", st['cell_left']),
            Paragraph("The error-correction code standardized in 3GPP TS 38.212 for large data blocks. Uses a sparse parity-check matrix H where H &middot; c<sup>T</sup> = 0.", st['cell_left']),
            Paragraph("<code>LDPC5GEncoder(k, n)</code> and <code>LDPC5GDecoder(encoder, num_iter=20)</code>.", st['cell_left'])
        ],
        [
            Paragraph("<b>Belief Propagation (BP)</b>", st['cell_left']),
            Paragraph("Iterative decoding algorithm where variable nodes and check nodes exchange soft probabilistic beliefs over 20 iterations.", st['cell_left']),
            Paragraph("Configured with <code>num_ldpc_iter=20</code> in <code>OFDMFECPipeline</code>.", st['cell_left'])
        ],
        [
            Paragraph("<b>Group Normalization (G=1)</b>", st['cell_left']),
            Paragraph("Normalizes features across channels per sample. Unlike BatchNorm, GroupNorm is completely independent of batch size (works perfectly at B=1).", st['cell_left']),
            Paragraph("<code>nn.GroupNorm(num_groups=1, num_channels=48)</code> in ConvBlock &amp; ResBlock.", st['cell_left'])
        ],
        [
            Paragraph("<b>Experience Replay Buffer</b>", st['cell_left']),
            Paragraph("Circular FIFO memory bank storing past frames. Randomly sampling mini-batches breaks temporal correlation and stops catastrophic forgetting.", st['cell_left']),
            Paragraph("<code>ExperienceReplayBuffer(capacity=128)</code> in <code>replay_buffer.py</code>.", st['cell_left'])
        ],
        [
            Paragraph("<b>Rotational Ambiguity (90&deg;)</b>", st['cell_left']),
            Paragraph("Square QAM constellations look identical when rotated by 90&deg;. Under blind fading, the receiver cannot tell orientation, causing BER &approx; 0.49.", st['cell_left']),
            Paragraph("Solved by inserting an ultra-sparse 2.1% pilot grid to anchor the constellation phase.", st['cell_left'])
        ]
    ]
    jt = Table(jargons_data, colWidths=[105, 235, 164])
    jt.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0f2b5c")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(jt)
    story.append(Spacer(1, 8))

    # ─── CHAPTER 4: EVERY EQUATION IN THE PAPER DECODED ───
    story.append(Paragraph("Chapter 4: Every Equation in the Paper Decoded", st['h1']))
    story.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor("#0f2b5c"), spaceBefore=1, spaceAfter=6))

    eqs = [
        ("Equation 1: Parity-Check Geometry",
         "c = LDPC-Encode(u),  subject to  H &middot; c<sup>T</sup> = 0 (mod 2)",
         "User bits u are mapped to codeword c such that multiplying by parity matrix H gives all zeros. "
         "This parity check acts as an oracle: if H &middot; c<sup>T</sup> = 0, the codeword is mathematically certified error-free."),

        ("Equation 2: The Received Multipath Signal",
         "Y[l, k] = H[l, k] &middot; X[l, k] + W[l, k]",
         "At symbol l and subcarrier k: received complex number Y equals transmitted symbol X multiplied by channel fading H, plus Gaussian noise W."),

        ("Equation 3: Noise Power Spectral Density (N0)",
         "N0 = Es / (M &middot; R &middot; 10<sup>0.1 &middot; (Eb/N0)<sub>dB</sub></sup>)",
         "Converts experimental SNR in dB into the exact noise variance added to the radio grid. Es=1 is average symbol energy, M=4 bits/symbol (16-QAM), R=0.5 is code rate."),

        ("Equation 4: Rayleigh Fading Probability Density Function",
         "p(|H|) = (|H| / &sigma;<sup>2</sup>) &middot; exp(-|H|<sup>2</sup> / (2&sigma;<sup>2</sup>))",
         "When signals bounce off hundreds of obstacles with no line-of-sight, the envelope amplitude |H| follows this Rayleigh distribution."),

        ("Equations 5 & 6: Classical Pilot Overhead and LMMSE Inversion",
         "&alpha;<sub>pilot</sub> = 2/14 &approx; 14.29%,   X&#770;<sub>LMMSE</sub> = (H&#770;* Y) / (|H&#770;|<sup>2</sup> + &sigma;<sub>e</sub><sup>2</sup> + N0)",
         "Classical baseline reserves symbols 2 and 11 for pilots (14.29% overhead). Inverts channel distortion using LMMSE equalization, which suffers noise amplification at low SNR."),

        ("Equations 7–10: The Spectral Efficiency Gain Formula",
         "&eta;<sub>Classical</sub> = 4 &times; 0.5 &times; (1 - 2/14) = 1.714 bits/s/Hz,   &eta;<sub>Proposed</sub> = 4 &times; 0.5 &times; (1 - 0) = 2.000 bits/s/Hz  &rarr;  +16.67% Gain",
         "Because our proposed neural receiver eliminates all pilots, every subcarrier carries user payload, yielding an unconditional +16.67% throughput improvement."),

        ("Equation 17: Exact Neural Parameter Count",
         "P<sub>total</sub> = 128,021 parameters &equiv; 0.488 MB (in float32)",
         "Analytical breakdown of weights in input projection, 3 residual blocks, GroupNorms, and output projection. Fits entirely in L2/L3 processor cache."),

        ("Equation 20: Disagreement Rate Metric (Reliability Filter)",
         "d = (1 / n) &sum; | Hard(&lambda;&#770;<sub>j</sub>) - c&#770;<sub>j</sub> |",
         "Measures the fraction of bits where the initial CNN demapper guess disagreed with the LDPC-corrected codeword. If d &le; 30%, the frame is accepted into the replay buffer."),

        ("Equation 22: Constellation Rotational Symmetry (Ambiguity)",
         "E[ (exp(j &pi;/2) &middot; s)<sup>4</sup> ] = E[ s<sup>4</sup> ]",
         "Rotating 16-QAM symbols by 90 degrees yields identical 4th-order statistics. Blind receivers cannot distinguish orientation in fading channels without a phase anchor.")
    ]

    for title, formula, desc in eqs:
        story.append(Paragraph(f"<b>{title}</b>", st['h2']))
        story.append(make_callout(f"<b>Formula:</b> {formula}<br/><b>Physical Meaning:</b> {desc}", st))
        story.append(Spacer(1, 3))
    story.append(Spacer(1, 8))

    # ─── CHAPTER 5: PROFESSOR DEFENSE MASTERCLASS (ORAL SCRIPT) ───
    story.append(Paragraph("Chapter 5: Professor Defense Masterclass (Top 7 Tough Questions & Exact Answers)", st['h1']))
    story.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor("#0f2b5c"), spaceBefore=1, spaceAfter=6))

    qa_list = [
        ("Q1: Why does the classical baseline fail at 6 dB while your neural receiver gets zero errors?",
         "<b>Your Defense:</b> 'At 6 dB low-SNR, Least Squares (LS) channel estimation on pilots suffers from severe noise enhancement. "
         "The noisy channel estimates distort the LMMSE equalizer, causing the classical APP demapper to produce compressed, miscalibrated LLRs. "
         "When fed into the LDPC decoder, error propagation causes a 65% frame error rate (BLER = 0.65). "
         "In contrast, our 2D ResCNN looks at the entire 14x64 grid jointly. Its 2D convolutional kernels perform spatial denoising across adjacent subcarriers and symbols, "
         "producing sharp, well-calibrated LLRs (91.37% raw bit accuracy). This falls well within the convergence basin of the LDPC decoder, achieving BER = 0.00000.'"),

        ("Q2: Isn't your self-supervised Delta loop just an artifact of good initial weights?",
         "<b>Your Defense:</b> 'No, professor. We explicitly proved this through our 4-way ablation study (Table V in our paper). "
         "When we froze the demapper weights (Delta OFF), the system suffered an error floor with post-FEC BER = 8.45 &times; 10<sup>-2</sup> at 6 dB. "
         "Activating the Delta loop (Delta ON) reduced demapper-decoder disagreement from 26.8% down to 0.1%, providing a +3.8 dB SNR gain and "
         "closing within 0.2 dB of the ideal genie-aided supervised bound without requiring any true labels.'"),

        ("Q3: Does this scale to real 5G NR block lengths beyond toy simulations?",
         "<b>Your Defense:</b> 'Yes, professor. We tested block lengths up to k = 8,400 information bits with 300 subcarriers (Table II), which represents a full 5 MHz 5G NR carrier "
         "and reaches the maximum structural ceiling of 3GPP LDPC Base Graph 1 (k_max = 8,448). "
         "As block length increases, classical baseline BLER worsens because codeword error probability accumulates with length (P_BLER = 1 - (1-Pb)^n). "
         "In contrast, the 2D ResCNN maintains BER = 0.00000 across all block lengths at 6 dB because the convolutional filters are grid-invariant.'"),

        ("Q4: Why did pure blind neural demapping fail in Rayleigh fading, and what is your solution?",
         "<b>Your Defense:</b> 'Pure blind demapping in Rayleigh multipath fading encounters the fundamental 90-degree rotational symmetry of square QAM constellations, "
         "where E[(exp(j &pi;/2)s)^4] = E[s^4]. Without a phase reference, the blind receiver cannot determine constellation orientation, causing bit flips and BER &approx; 0.49. "
         "We resolved this by introducing an ultra-sparse pilot grid of only 2 subcarriers per slot (2.1% overhead). "
         "This provides the necessary phase anchor to break ambiguity, achieving BER = 0.00000 while preserving over 95% of the spectral efficiency advantage over classical 14.3% pilots.'"),

        ("Q5: Is this practical for real-time edge hardware? What is the latency?",
         "<b>Your Defense:</b> 'Yes, professor. We profiled the system empirically on standard CPU hardware (Table VII):<br/>"
         "1. The 2D ResCNN has only 128,021 parameters (0.488 MB memory), fitting inside the L2/L3 cache of modern baseband chips.<br/>"
         "2. Batched inference latency is 0.98 ms per frame for k = 1,792 bits, which satisfies the standard 5G NR 1.0 ms subframe timing requirement.<br/>"
         "3. The online Delta adaptation step (&approx; 300 ms) runs asynchronously in a low-priority background thread once every few hundred slots, adding zero latency to real-time packet forwarding.'"),

        ("Q6: How does the Experience Replay Buffer prevent catastrophic forgetting?",
         "<b>Your Defense:</b> 'If the wireless channel suddenly transitions from high SNR (16 dB) to low SNR (6 dB), a naive online gradient update would immediately overfit to 6 dB noise "
         "and forget how to demodulate clean 16 dB signals. By sampling mini-batches uniformly from a 128-frame circular buffer of past valid transmissions, "
         "the network maintains a balanced distribution across varying noise levels, guaranteeing gradient stability.'"),

        ("Q7: What is the exact spectral efficiency advantage?",
         "<b>Your Defense:</b> 'Removing all Kronecker pilots gives an exact +16.67% increase in data throughput (2.000 vs 1.714 bits/s/Hz). "
         "In terms of payload capacity, this provides an additional +256 bits per slot at k = 1,792 and +1,200 bits per slot at k = 8,400 with zero error penalty.'" )
    ]

    for q, ans in qa_list:
        story.append(Paragraph(f"<b>{q}</b>", st['h2']))
        story.append(make_callout(ans, st, bg_color="#f1f5f9", border_color="#94a3b8"))
        story.append(Spacer(1, 4))
    story.append(Spacer(1, 8))

    # ─── CHAPTER 6: KEY METRICS CHEAT SHEET ───
    story.append(Paragraph("Chapter 6: Key Metrics & Golden Numbers to Memorize (Cheat Sheet)", st['h1']))
    story.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor("#0f2b5c"), spaceBefore=1, spaceAfter=6))

    cheat_sheet = [
        [Paragraph("<b>Metric / Parameter</b>", st['header']), Paragraph("<b>Value</b>", st['header']), Paragraph("<b>Significance to Quote to Professor</b>", st['header'])],
        [Paragraph("<b>Spectral Efficiency Gain</b>", st['cell_left']), Paragraph("<b>+16.67%</b>", st['cell']), Paragraph("2.000 vs 1.714 bits/s/Hz (100% data payload, 0% pilots).", st['cell_left'])],
        [Paragraph("<b>Low-SNR Waterfall Gain</b>", st['cell_left']), Paragraph("<b>+2.0 to +3.8 dB</b>", st['cell']), Paragraph("Reaches BER = 0.00000 at 6 dB where classical baseline BLER is 65%.", st['cell_left'])],
        [Paragraph("<b>Model Parameter Count</b>", st['cell_left']), Paragraph("<b>128,021 weights</b>", st['cell']), Paragraph("Only 0.488 MB memory in float32. Fits in L2/L3 hardware cache.", st['cell_left'])],
        [Paragraph("<b>Batched Inference Latency</b>", st['cell_left']), Paragraph("<b>0.98 ms / frame</b>", st['cell']), Paragraph("Meets the standard 3GPP 5G NR 1.0 ms slot timing constraint on CPU.", st['cell_left'])],
        [Paragraph("<b>Max Tested Block Length</b>", st['cell_left']), Paragraph("<b>k = 8,400 bits</b>", st['cell']), Paragraph("Reaches 3GPP LDPC Base Graph 1 structural ceiling (Nsc = 300, 5 MHz).", st['cell_left'])],
        [Paragraph("<b>Delta Disagreement Reduction</b>", st['cell_left']), Paragraph("<b>26.8% &rarr; 0.1%</b>", st['cell']), Paragraph("Proves online adaptation is active and closes within 0.2 dB of supervised.", st['cell_left'])],
        [Paragraph("<b>Rotational Ambiguity Solution</b>", st['cell_left']), Paragraph("<b>2.1% sparse pilot</b>", st['cell']), Paragraph("Anchors phase in Rayleigh fading while keeping &gt;95% of throughput gain.", st['cell_left'])],
        [Paragraph("<b>Payload Image Recovery</b>", st['cell_left']), Paragraph("<b>Infinite PSNR</b>", st['cell']), Paragraph("Bit-for-bit perfect visual reconstruction at 8 dB; 38.5 dB PSNR at 6 dB.", st['cell_left'])]
    ]
    cst = Table(cheat_sheet, colWidths=[120, 95, 289])
    cst.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0f2b5c")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(cst)

    # Build PDF
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[OK] Master Defense Guide PDF successfully built at: {pdf_path}")

    # Build Companion Markdown Document
    md_content = """# Master Project Understanding & Oral Defense Guide
## 5G NR Joint Neural Demodulation and LDPC Decoding: Theory, Equations & Defense Script

---

### Executive Overview & Key Specs

| Parameter / Metric | Specification / Value |
|:---|:---|
| **Core Architecture** | Pilot-Free 2D ResCNN Demapper + 5G NR LDPC Belief Propagation Decoder |
| **Modulation & Code** | 16-QAM ($M=4$ bits/symbol), 3GPP LDPC Base Graph 1 ($R=0.5$) |
| **Model Size** | **128,021 trainable weights** (0.488 MB in float32) |
| **Spectral Efficiency Gain** | **+16.67%** ($2.000$ vs. $1.714\text{ bits/s/Hz}$) |
| **Coding Advantage** | **+2.0 dB to +3.8 dB** waterfall gain over classical 3GPP baseline |
| **Inference Latency** | **0.98 ms / frame** (batched CPU, within 1.0 ms 5G slot timing) |
| **Maximum Block Length** | $k = 8,400$ bits ($N_{\text{sc}} = 300$, 5 MHz channel, 3GPP LDPC structural ceiling) |
| **Channel Conditions** | Additive White Gaussian Noise (AWGN) & Dispersive Rayleigh Multipath Fading |

---

## Chapter 1: The Core Story in Plain English (The 2-Minute Pitch)

1. **The Real-World Wireless Problem (The Pilot Tax):**
   In standardized 5G New Radio (NR), whenever a mobile phone or base station transmits data across radio waves, it is forced to waste **14.29% of its total bandwidth** on fixed benchmark signals called *Pilots* (Demodulation Reference Signals, DMRS). Pilots carry zero user information—they exist solely so the receiver can estimate how the wireless channel distorted the signal. Worse yet, in low-signal environments (low SNR), estimating the channel using noisy pilots causes severe errors that corrupt the entire packet.

2. **Our Invention (Pilot-Free Joint Neural Demapping):**
   We completely eliminate all pilots (100% data transmission), delivering an immediate, permanent **+16.67% spectral efficiency gain**. Instead of relying on fragile channel estimation formulas, we deploy a lightweight **2D Residual Convolutional Neural Network (2D ResCNN)**. The CNN processes the entire time-frequency resource grid directly like a 2D image, capturing complex spatial noise correlations and outputting soft bit probabilities.

3. **The Self-Supervised Magic (The Teacher-Student Delta Loop):**
   How can the CNN learn online without any pilots or ground-truth bit labels? We pair the CNN with a standardized **5G NR LDPC Belief Propagation Decoder**. Think of the CNN as an *eager student* and the LDPC decoder as a *strict mathematical spell-checker*:
   - The student (CNN) makes an initial guess of the bits (which contains 10%–20% raw errors).
   - The LDPC decoder uses strict parity-check geometry ($\mathbf{H} \cdot \mathbf{c}^T = \mathbf{0}$) to correct these errors, producing certified mathematical truth.
   - We compute the difference (the **Delta signal $\Delta$**) between what the CNN guessed and what the LDPC decoder proved.
   - The CNN updates its weights using this Delta signal through an Experience Replay Buffer—**learning continuously online without true labels!**

---

## Chapter 2: The End-to-End Pipeline Step-by-Step

- **Stage 1 (TX - Payload):** Information bits $\mathbf{u} \in \{0, 1\}^k$ are generated ($k = 1,792$ up to $8,400$ bits).
- **Stage 2 (TX - FEC Encode):** 5G NR LDPC BG1 encodes $\mathbf{u}$ into codeword $\mathbf{c}$ of length $n = k/R$ ($n = 3,584$ to $16,800$ bits).
- **Stage 3 (TX - Modulation):** 16-QAM maps codeword $\mathbf{c}$ into complex symbols $\mathbf{s}$ (4 bits/symbol).
- **Stage 4 (TX - Grid Mapping):** Mapped into a $14 \times N_{\text{sc}}$ Resource Grid with **100% data and 0% pilots**. IFFT + CP generates time-domain signal.
- **Stage 5 (Channel):** Propagates across AWGN or Rayleigh multipath fading with noise power $N_0$.
- **Stage 6 (RX - Grid Reconstruction):** CP removal + FFT reconstructs the 2D grid $\mathbf{Y}$ formatted as 2 real channels $[\Re(\mathbf{Y}), \Im(\mathbf{Y})]$.
- **Stage 7 (RX - Neural Demapping):** 2D ResCNN outputs calibrated soft Log-Likelihood Ratios (LLRs) $\hat{\boldsymbol{\lambda}}$.
- **Stage 8 (RX - LDPC Decoding):** Belief Propagation decoder runs 20 message-passing iterations to output user bits $\hat{\mathbf{u}}$.
- **Stage 9 (Adaptation - Delta Loop):** Re-encodes $\hat{\mathbf{c}} = \text{LDPC-Encode}(\hat{\mathbf{u}})$. If disagreement $d \le 30\%$, pushes to Replay Buffer and updates weights via AdamW.

---

## Chapter 3: Key Mathematical Equations Decoded

- **Equation 1 (Parity Geometry):** $\mathbf{H}_{\text{LDPC}} \mathbf{c}^T = \mathbf{0} \pmod 2$. Proves codeword validity without external labels.
- **Equation 2 (Received Signal):** $Y[l, k] = H[l, k] X[l, k] + W[l, k]$.
- **Equation 3 (Noise PSD):** $N_0 = \frac{E_s}{M \cdot R \cdot 10^{0.1 \cdot (E_b/N_0)_{\text{dB}}}}$. Converts SNR in dB to exact noise variance.
- **Equation 4 (Rayleigh Envelope):** $p(|H|) = \frac{|H|}{\sigma^2} e^{-|H|^2 / (2\sigma^2)}$.
- **Equation 5 (Pilot Overhead):** $\alpha_{\text{pilot}} = \frac{2}{14} \approx 14.29\%$ in classical 3GPP Kronecker patterns.
- **Equations 7–10 (Spectral Efficiency Gain):** $\eta_{\text{Classical}} = 1.714\text{ bits/s/Hz}$ vs. $\eta_{\text{Proposed}} = 2.000\text{ bits/s/Hz}$ $\to$ **$+16.67\%$ Net Gain**.
- **Equation 17 (Parameter Count):** Exactly **128,021 parameters** ($0.488\text{ MB}$ memory footprint).
- **Equation 20 (Reliability Disagreement):** $d = \frac{1}{n} \sum |\text{Hard}(\hat{\lambda}) - \hat{c}|$. Replay buffer gating threshold set to $\tau = 0.30$.
- **Equation 22 (Rotational Ambiguity):** $\mathbb{E}[(e^{j\pi/2} s)^4] = \mathbb{E}[s^4]$. Explains why pure blind demapping requires an ultra-sparse 2.1% pilot phase anchor.

---

## Chapter 4: Professor Defense Script (Top 7 Tough Questions)

### Q1: Why does the classical baseline fail at 6 dB while your neural receiver gets zero errors?
> **Answer:** At 6 dB, Least Squares channel estimation on pilots suffers from noise enhancement. The noisy estimates distort LMMSE equalization, causing the classical APP demapper to produce compressed, miscalibrated LLRs that fail the LDPC decoder on 65% of frames ($\text{BLER}=0.65$). The 2D ResCNN looks at the entire grid jointly, performing spatial denoising across adjacent subcarriers and symbols to produce sharp LLRs (91.37% raw accuracy), easily achieving $\text{BER}=0.00000$.

### Q2: Isn't your self-supervised Delta loop just an artifact of good initial weights?
> **Answer:** No. Our 4-way ablation (Table V) proves that freezing weights (Delta OFF) leaves a permanent error floor ($\text{BER}=8.45 \times 10^{-2}$ at 6 dB). Turning the Delta loop ON reduces demapper-decoder disagreement from 26.8% to 0.1%, yielding a $+3.8\text{ dB}$ gain and operating within $0.2\text{ dB}$ of the ideal supervised upper bound without ground-truth labels.

### Q3: Does this scale to real 5G NR block lengths beyond toy simulations?
> **Answer:** Yes. We evaluated block lengths up to $k = 8,400$ bits ($N_{\text{sc}} = 300$, 5 MHz channel), which operates at the structural ceiling of 3GPP LDPC Base Graph 1 ($k_{\max} = 8,448$). While classical BLER degrades at large block lengths, the 2D ResCNN maintains $\text{BER}=0.00000$ across all block sizes.

### Q4: Why did pure blind neural demapping fail in Rayleigh fading, and what is your solution?
> **Answer:** Pure blind demapping encounters the fundamental $90^\circ$ rotational symmetry of square QAM constellations, where $\mathbb{E}[(e^{j\pi/2}s)^4] = \mathbb{E}[s^4]$, causing $\text{BER}\approx 0.49$. We resolved this with an ultra-sparse pilot grid of only 2 subcarriers per slot (2.1% overhead), which anchors the phase and achieves $\text{BER}=0.00000$ while preserving over 95% of our throughput advantage.

### Q5: Is this practical for real-time edge hardware?
> **Answer:** Yes. The model has only 128,021 parameters ($0.488\text{ MB}$ footprint) and runs in **$0.98\text{ ms}$ per frame** on CPU, satisfying the 5G NR $1.0\text{ ms}$ slot constraint. The online training step runs in a background thread, adding zero latency to packet forwarding.

---

## Chapter 5: Key Numbers Cheat Sheet to Memorize

- **Spectral Efficiency Gain:** $+16.67\%$ ($2.000$ vs. $1.714\text{ bits/s/Hz}$)
- **Coding Gain:** $+2.0\text{ dB}$ to $+3.8\text{ dB}$ waterfall advantage over 3GPP baseline
- **Model Size:** $128,021$ parameters ($0.488\text{ MB}$ memory footprint)
- **CPU Inference Latency:** $0.98\text{ ms}$ (batched, 5G slot compliant)
- **Max Block Size:** $k = 8,400$ bits ($N_{\text{sc}} = 300$, 3GPP LDPC BG1 ceiling)
- **Disagreement Drop:** $26.8\% \to 0.1\%$ via self-supervised Delta loop
- **Phase Anchor:** $2.1\%$ sparse pilot resolves $90^\circ$ rotational ambiguity in Rayleigh fading
- **Image Payload Recovery:** Infinite PSNR at $8\text{ dB}$, near-perfect ($38.5\text{ dB}$ PSNR) at $6\text{ dB}$
"""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[OK] Master Defense Guide Markdown successfully built at: {md_path}")

if __name__ == "__main__":
    pdf_out = r"c:\Users\hamhamai001\Desktop\Amr using EL\Master_Project_Understanding_and_Defense_Guide.pdf"
    md_out = r"c:\Users\hamhamai001\Desktop\Amr using EL\Master_Project_Understanding_and_Defense_Guide.md"
    build_defense_guide(pdf_out, md_out)
