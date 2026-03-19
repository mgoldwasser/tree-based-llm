"""
Build paper PDF from results data and figures using reportlab.
No LaTeX installation required.

Usage: python build_pdf.py
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, KeepTogether
)
from reportlab.lib import colors

OUTPUT = os.path.join(os.path.dirname(__file__), "paper.pdf")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")


def build():
    doc = SimpleDocTemplate(OUTPUT, pagesize=letter,
                            leftMargin=1*inch, rightMargin=1*inch,
                            topMargin=0.8*inch, bottomMargin=0.8*inch)

    styles = getSampleStyleSheet()

    # Custom styles
    styles.add(ParagraphStyle('PaperTitle', parent=styles['Title'],
                              fontSize=17, leading=22, spaceAfter=6))
    styles.add(ParagraphStyle('Author', parent=styles['Normal'],
                              fontSize=12, alignment=TA_CENTER, spaceAfter=18))
    styles.add(ParagraphStyle('Abstract', parent=styles['Normal'],
                              fontSize=10, leading=14, leftIndent=36,
                              rightIndent=36, spaceAfter=12, alignment=TA_JUSTIFY))
    styles.add(ParagraphStyle('SectionHead', parent=styles['Heading1'],
                              fontSize=14, spaceBefore=18, spaceAfter=8))
    styles.add(ParagraphStyle('SubsectionHead', parent=styles['Heading2'],
                              fontSize=12, spaceBefore=14, spaceAfter=6))
    styles.add(ParagraphStyle('Body', parent=styles['Normal'],
                              fontSize=10.5, leading=14, alignment=TA_JUSTIFY,
                              spaceAfter=6))
    styles.add(ParagraphStyle('Caption', parent=styles['Normal'],
                              fontSize=9.5, leading=12, alignment=TA_CENTER,
                              spaceAfter=12, textColor=HexColor('#333333')))
    styles.add(ParagraphStyle('Equation', parent=styles['Normal'],
                              fontSize=10.5, alignment=TA_CENTER, spaceAfter=8,
                              spaceBefore=8, fontName='Courier'))
    styles.add(ParagraphStyle('BibItem', parent=styles['Normal'],
                              fontSize=9, leading=11, leftIndent=24,
                              firstLineIndent=-24, spaceAfter=4))

    story = []

    def sec(title):
        story.append(Paragraph(title, styles['SectionHead']))

    def subsec(title):
        story.append(Paragraph(title, styles['SubsectionHead']))

    def body(text):
        story.append(Paragraph(text, styles['Body']))

    def eq(text):
        story.append(Paragraph(f"<font face='Courier'>{text}</font>", styles['Equation']))

    def space(h=6):
        story.append(Spacer(1, h))

    def fig(filename, caption, width=6.5*inch):
        path = os.path.join(FIG_DIR, filename)
        if os.path.exists(path):
            img = Image(path, width=width, height=width*0.6)
            story.append(img)
            story.append(Paragraph(caption, styles['Caption']))

    def table_block(data, col_widths=None):
        t = Table(data, colWidths=col_widths)
        t.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9.5),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('LINEABOVE', (0, 0), (-1, 0), 1, colors.black),
            ('LINEBELOW', (0, 0), (-1, 0), 0.5, colors.black),
            ('LINEBELOW', (0, -1), (-1, -1), 1, colors.black),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(t)
        space(8)

    # ── Title ────────────────────────────────────────────────────────────
    story.append(Paragraph(
        "Tree-Based Attention: Replacing Linear Projections<br/>"
        "with Differentiable Decision Forests in Transformers",
        styles['PaperTitle']))
    story.append(Paragraph("Matt Goldwasser", styles['Author']))

    # ── Abstract ─────────────────────────────────────────────────────────
    story.append(Paragraph("<b>Abstract</b>", styles['SubsectionHead']))
    body("We investigate replacing the dense linear projections (Q, K, V) in transformer attention "
         "with differentiable soft decision trees, trained end-to-end via backpropagation. We introduce "
         "<i>BatchedTreeForest</i>, an efficient implementation that computes all trees in a single batched "
         "einsum operation, and <i>BoostedForest</i>, a gradient-boosting-inspired architecture that combines "
         "a linear base projection with tree-based residual corrections via feature pass-through. On "
         "character-level language modeling (Shakespeare), our Boosted Forest transformer achieves "
         "<b>41.1% validation accuracy</b>, matching the standard transformer's <b>41.3%</b> &mdash; while "
         "using tree-based projections for all Q/K/V/O attention computations. We analyze the failure mode of "
         "pure tree-based projections (premature entropy collapse), the critical role of architectural choices "
         "(initialization, optimizer groups, QKV fusion), and present a roadmap for scaling tree-based attention "
         "to larger models.")

    # ── 1. Introduction ──────────────────────────────────────────────────
    sec("1. Introduction")
    body("The transformer architecture (Vaswani et al., 2017) relies fundamentally on linear projections "
         "to compute queries, keys, and values for attention. These projections are simple matrix "
         "multiplications &mdash; computationally efficient but limited to learning linear relationships "
         "between input features.")
    body("Decision trees offer a compelling alternative: they learn piecewise-constant functions through "
         "hierarchical feature partitioning, can capture non-linear relationships naturally, and provide "
         "interpretable routing decisions. However, classical decision trees use hard (non-differentiable) "
         "splits, making them incompatible with gradient-based training.")
    body("<i>Soft decision trees</i> (Irsoy et al., 2012; Frosst &amp; Hinton, 2017) resolve this by "
         "replacing hard splits with sigmoid-gated routing, allowing gradients to flow through all tree "
         "paths simultaneously. We build on this foundation to create tree-based projection layers that "
         "serve as drop-in replacements for linear layers in transformer attention.")
    body("Our key contributions: (1) <b>BatchedTreeForest</b>: a batched tensor implementation computing all "
         "trees via a single einsum, achieving 4&times; more trees at similar speed. (2) <b>BoostedForest</b>: "
         "a multi-stage architecture augmenting a linear base with tree-based residual corrections. "
         "(3) Comprehensive analysis of failure modes and architectural choices. "
         "(4) Empirical validation on Shakespeare, matching standard transformer performance.")

    # ── 2. Background ────────────────────────────────────────────────────
    sec("2. Background")
    subsec("2.1 Soft Decision Trees")
    body("A soft decision tree of depth D has 2<super>D</super> &minus; 1 internal nodes and "
         "2<super>D</super> leaves. Each internal node <i>i</i> computes a soft routing probability:")
    eq("p_left(x) = sigmoid((w_i &middot; x + b_i) / &tau;)")
    body("The probability of reaching leaf <i>l</i> is the product of routing decisions along the path. "
         "The tree output is a probability-weighted sum over learned leaf vectors. This formulation "
         "is fully differentiable with respect to all parameters.")

    subsec("2.2 Transformer Attention")
    body("Standard multi-head attention computes Q = xW<sub>Q</sub>, K = xW<sub>K</sub>, "
         "V = xW<sub>V</sub>, then Attn(Q,K,V) = softmax(QK<super>T</super>/&radic;d<sub>k</sub>)V. "
         "We replace the linear projections W<sub>Q</sub>, W<sub>K</sub>, W<sub>V</sub> with tree-based "
         "projection layers.")

    # ── 3. Method ────────────────────────────────────────────────────────
    sec("3. Method")
    subsec("3.1 BatchedTreeForest")
    body("To avoid iterating over individual trees in Python, we store all tree parameters in stacked "
         "tensors: decision weights (T, N<sub>internal</sub>, D<sub>in</sub>), leaf outputs "
         "(T, N<sub>leaves</sub>, D<sub>out</sub>), and mixture weights (T,). The routing computation "
         "for all T trees is a single batched einsum:")
    eq("decisions = einsum('bsd,tnd-&gt;bstn', x, W_decision)")
    body("Leaf probabilities are computed in log-space to prevent numerical underflow through deep trees. "
         "The final output is a softmax-weighted mixture of per-tree outputs.")

    subsec("3.2 BoostedForest")
    body("Inspired by gradient boosting, we compose a linear base projection with multiple tree-based "
         "correction stages:")
    eq("output = W_base &middot; x + b + &Sigma; &gamma;_s &middot; Forest_s([x; output_{s-1}])")
    body("where &gamma;<sub>s</sub> is a learned shrinkage factor (initialized to 0.1) and "
         "[x; output<sub>s-1</sub>] denotes <i>feature pass-through</i> &mdash; concatenating the original "
         "input with the current running output. This allows each stage to observe both raw features and "
         "what previous stages predicted, enabling residual corrections.")

    subsec("3.3 QKV Fusion")
    body("Instead of three independent forests for Q, K, V, we use a single forest with tripled output "
         "dimension: [Q; K; V] = Forest<sub>QKV</sub>(x). This shares routing decisions across Q, K, V, "
         "reducing routing computation by approximately 3&times;.")

    subsec("3.4 Tree-Aware Training")
    body("<b>Initialization:</b> We use N(0, 0.02) for decision weights (Xavier is too large, causing "
         "sigmoid saturation). <b>Optimizer groups:</b> Decision weights get 3&times; learning rate and "
         "zero weight decay. <b>Entropy regularization:</b> Penalizes binary entropy of routing decisions "
         "to encourage crisp splits. <b>Temperature annealing:</b> Cosine schedule from &tau;=1.0 to &tau;=0.1.")

    # ── 4. Experiments ───────────────────────────────────────────────────
    sec("4. Experiments")
    subsec("4.1 Setup")
    body("We evaluate on character-level language modeling using the Tiny Shakespeare dataset (Karpathy, 2015): "
         "1.1M characters, 65-character vocabulary, 90/10 train/val split. Architecture: 4 transformer layers, "
         "d<sub>model</sub>=128, 4 attention heads, sequence length 256, batch size 32, 2000 training steps.")
    space(4)

    table_block([
        ['Model', 'Description', 'Params'],
        ['Standard', 'Linear Q/K/V/O projections', '843K'],
        ['Batched Forest', '12 trees, depth 3, QKV-fused', '862K'],
        ['Boosted Forest', 'Linear base + 3 stages \u00d7 12 trees \u00d7 depth 2', '1.5M'],
    ], col_widths=[1.5*inch, 3*inch, 0.8*inch])

    subsec("4.2 Results on Shakespeare")

    fig('main_results.png',
        '<b>Figure 1:</b> Main results on Shakespeare character-level LM (2000 steps, CPU). '
        '(a) Validation accuracy vs. steps. (b) Validation loss vs. steps. '
        '(c) Routing entropy collapse. (d) Convergence vs. wall-clock time.')

    table_block([
        ['Model', 'Val Accuracy', 'Val Loss', 'ms/step', 'Slowdown'],
        ['Standard Transformer', '41.3%', '1.986', '726', '1.0\u00d7'],
        ['Batched Forest (attn)', '27.4%', '2.471', '1,611', '2.2\u00d7'],
        ['Boosted Forest (attn)', '41.1%', '1.987', '3,042', '4.2\u00d7'],
    ], col_widths=[1.8*inch, 1*inch, 0.8*inch, 0.8*inch, 0.8*inch])
    body("<b>Table 2:</b> Shakespeare results after 2000 training steps.")
    space(4)

    body("The Boosted Forest achieves validation accuracy within 0.2 percentage points of the standard "
         "transformer, demonstrating that tree-based projections can match linear projections on real "
         "language data when properly configured.")

    subsec("4.3 Failure Mode: Premature Entropy Collapse")
    body("The pure Batched Forest plateaus at 27.4% accuracy. Routing entropy drops from 0.69 to 0.005 "
         "over training (Figure 1c), meaning soft trees become effectively hard within the first few "
         "hundred steps. Once routing hardens, trees cannot adapt their feature partitioning and learning stalls.")
    body("The Boosted Forest exhibits the same entropy collapse but does not suffer accuracy degradation. "
         "The linear base projection continues to learn throughout training regardless of tree routing entropy. "
         "Tree stages provide corrections early when routing is soft, and learned shrinkage factors regulate "
         "tree influence after hardening.")

    subsec("4.4 Speed Analysis")
    fig('speed_comparison.png',
        '<b>Figure 2:</b> Training speed and model size. QKV fusion reduces tree overhead from ~8\u00d7 to ~4\u00d7.')

    body("QKV fusion reduces attention projection from 4 separate forest passes to 2 (fused QKV + output). "
         "The remaining 4.2\u00d7 gap comes from the depth loop in leaf probability computation and "
         "three sequential boosted stages with feature pass-through. On GPU, we estimate the gap would "
         "narrow to approximately 1.5&ndash;2\u00d7.")

    # ── 5. Analysis ──────────────────────────────────────────────────────
    sec("5. Analysis and Discussion")
    subsec("5.1 Why Boosted Forest Works")
    body("Three design decisions drive success: (1) The <b>linear base provides a guaranteed gradient path</b> "
         "&mdash; even if tree routing collapses, the linear projection keeps learning. "
         "(2) <b>Residual pass-through enables correction learning</b> &mdash; each stage sees both raw features "
         "and current predictions, focusing corrections where the base is weakest. "
         "(3) <b>Learned shrinkage controls tree influence</b> &mdash; the model automatically reduces tree "
         "influence if they are not contributing useful corrections.")

    subsec("5.2 Why Pure Tree Projections Fail")
    body("The Batched Forest's poor performance stems from a conflict between temperature annealing and "
         "optimization dynamics. As temperature drops, routing sharpens based on patterns from limited early "
         "data. Once hard, partition boundaries freeze and recovery from suboptimal early routing is impossible. "
         "This is analogous to the 'rich get richer' problem in mixture models.")

    subsec("5.3 Implications")
    body("Our results suggest trees are most effective as <i>corrections to strong base models</i> rather "
         "than standalone replacements. This aligns with gradient boosting literature. The pattern &mdash; "
         "linear base + tree corrections &mdash; may generalize beyond attention to any component where "
         "non-linear refinement of a linear operation is desired.")

    subsec("5.4 Limitations")
    body("(1) <b>Speed:</b> 4.2\u00d7 CPU slowdown needs GPU optimization. "
         "(2) <b>Scale:</b> Only tested at 843K&ndash;1.5M parameters. "
         "(3) <b>Temperature:</b> Cosine schedule not optimized; entropy collapses too fast. "
         "(4) <b>Interpretability:</b> Entropy collapse makes late-training routing fixed rather than "
         "meaningfully interpretable.")

    # ── 6. Related Work ──────────────────────────────────────────────────
    sec("6. Related Work")
    body("<b>Soft Decision Trees:</b> Irsoy et al. (2012) introduced soft trees. Frosst &amp; Hinton (2017) "
         "used them for distillation. Hazimeh et al. (2020) proposed differentiable trees for tabular data.")
    body("<b>Tree-based Neural Networks:</b> Deep Neural Decision Forests (Kontschieder et al., 2015) "
         "combined forests with neural features. Adaptive Neural Trees (Tanno et al., 2019) learned "
         "tree structure alongside parameters.")
    body("<b>Mixture of Experts:</b> BoostedForest shares conceptual similarity with MoE (Shazeer et al., 2017), "
         "where tree routing serves as continuous expert selection.")
    body("<b>Efficient Attention:</b> Low-rank (Wang et al., 2020), sparse (Child et al., 2019), and "
         "kernel-based (Katharopoulos et al., 2020) methods offer alternative projections. Tree-based "
         "projections provide a distinct inductive bias &mdash; piecewise-constant feature partitioning.")

    # ── 7. Future Work ───────────────────────────────────────────────────
    sec("7. Future Work")
    body("(1) <b>Shared routing:</b> All trees share routing but have separate leaf outputs, reducing "
         "routing computation by T\u00d7. "
         "(2) <b>Conditional tree selection (MoTE):</b> Route tokens to top-K trees, reducing cost. "
         "(3) <b>GPU optimization:</b> Mixed precision and torch.compile to close the speed gap. "
         "(4) <b>Larger scale:</b> Evaluate on larger models and datasets. "
         "(5) <b>Routing analysis:</b> Investigate what linguistic features routing learns to partition on.")

    # ── 8. Conclusion ────────────────────────────────────────────────────
    sec("8. Conclusion")
    body("We demonstrate that differentiable decision trees can serve as effective projection layers in "
         "transformer attention, matching standard linear projections on character-level Shakespeare "
         "language modeling (41.1% vs. 41.3% validation accuracy). The key insight is that trees work "
         "best as <i>residual corrections to a linear base</i> &mdash; not as standalone replacements. "
         "This Boosted Forest architecture provides a robust gradient path through the linear base while "
         "allowing trees to learn complementary non-linear patterns.")
    body("While the current implementation incurs a 4.2\u00d7 speed penalty on CPU, the approach opens "
         "a new direction in transformer design: hybrid architectures combining the efficiency of linear "
         "projections with the expressive power of learned feature partitioning.")

    # ── References ───────────────────────────────────────────────────────
    sec("References")
    refs = [
        "Child, R., Gray, S., Radford, A., &amp; Sutskever, I. (2019). Generating long sequences with sparse transformers. <i>arXiv:1904.10509</i>.",
        "Frosst, N. &amp; Hinton, G. (2017). Distilling a neural network into a soft decision tree. <i>CEx Workshop, AI*IA</i>.",
        "Hazimeh, H., et al. (2020). The tree ensemble layer: Differentiability meets conditional computation. <i>ICML</i>.",
        "Irsoy, O., Yildiz, O.T., &amp; Alpaydin, E. (2012). Soft decision trees. <i>ICPR</i>.",
        "Karpathy, A. (2015). The unreasonable effectiveness of recurrent neural networks. <i>Blog post</i>.",
        "Katharopoulos, A., et al. (2020). Transformers are RNNs: Fast autoregressive transformers with linear attention. <i>ICML</i>.",
        "Kontschieder, P., et al. (2015). Deep neural decision forests. <i>ICCV</i>.",
        "Shazeer, N., et al. (2017). Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. <i>ICLR</i>.",
        "Tanno, R., et al. (2019). Adaptive neural trees. <i>ICML</i>.",
        "Vaswani, A., et al. (2017). Attention is all you need. <i>NeurIPS</i>.",
        "Wang, S., et al. (2020). Linformer: Self-attention with linear complexity. <i>arXiv:2006.04768</i>.",
    ]
    for ref in refs:
        story.append(Paragraph(ref, styles['BibItem']))

    # Build
    doc.build(story)
    print(f"PDF saved to {OUTPUT}")


if __name__ == "__main__":
    build()
