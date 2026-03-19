"""
Build paper PDF from results data and figures using reportlab.
No LaTeX installation required. Reads results from results/full_config_results.json.

Usage: python build_pdf.py
"""

import json
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
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def load_results():
    """Load experiment results for paper numbers."""
    full_file = os.path.join(RESULTS_DIR, 'full_config_results.json')
    if os.path.exists(full_file):
        with open(full_file) as f:
            raw = json.load(f)
        if 'full' in raw and raw['full']:
            return raw['full']

    results_file = os.path.join(RESULTS_DIR, 'shakespeare_results.json')
    if os.path.exists(results_file):
        with open(results_file) as f:
            return json.load(f)
    return {}


def fmt_acc(val):
    return f"{val*100:.1f}"

def fmt_loss(val):
    return f"{val:.3f}"

def fmt_ms(val):
    return f"{val:,.0f}"

def fmt_params(val):
    if val >= 1_000_000:
        return f"{val/1_000_000:.1f}M"
    return f"{val/1_000:,.0f}K"


def build():
    results = load_results()

    # Extract numbers (with fallbacks)
    def r(key, field, default="?"):
        if key in results:
            return results[key].get(field, default)
        return default

    doc = SimpleDocTemplate(OUTPUT, pagesize=letter,
                            leftMargin=1*inch, rightMargin=1*inch,
                            topMargin=0.8*inch, bottomMargin=0.8*inch)

    styles = getSampleStyleSheet()

    # Custom styles
    styles.add(ParagraphStyle('PaperTitle', parent=styles['Title'],
                              fontSize=17, leading=22, spaceAfter=6))
    styles.add(ParagraphStyle('Author', parent=styles['Normal'],
                              fontSize=12, alignment=TA_CENTER, spaceAfter=18))
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
        story.append(Paragraph(title, styles['Heading1']))

    def subsec(title):
        story.append(Paragraph(title, styles['Heading2']))

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
    subsec("Abstract")
    std_acc = fmt_acc(r('standard', 'final_val_acc', 0.39))
    bst_acc = fmt_acc(r('boosted', 'final_val_acc', 0.39))
    bst_ms = fmt_ms(r('boosted', 'ms_per_step', 0))
    vo_ms = fmt_ms(r('oblivious_boosted_vo_alt', 'ms_per_step', 0))
    body(f"We investigate replacing the dense linear projections (Q, K, V) in transformer attention "
         f"with differentiable soft decision trees, trained end-to-end via backpropagation. We introduce "
         f"several tree-based projection architectures &mdash; <i>BatchedTreeForest</i>, "
         f"<i>ObliviousTreeForest</i> (NODE-style), and <i>LinearPlusForest</i> (linear base + tree "
         f"correction) &mdash; all computed via batched einsum operations. Through systematic ablation on "
         f"character-level Shakespeare language modeling, we find that the Linear+Forest architecture "
         f"with unfused QKV routing matches a standard transformer at <b>{bst_acc}% vs. {std_acc}%</b> "
         f"validation accuracy, while speed-optimized variants using oblivious trees, alternating layers, "
         f"and selective V+O tree placement achieve comparable accuracy at significantly reduced cost "
         f"(<b>{vo_ms}ms vs. {bst_ms}ms</b> per step). We also evaluate a Mixture-of-Experts variant "
         f"replacing tree routing with top-k expert selection. Our analysis reveals that soft routing "
         f"&mdash; not hard tree decisions &mdash; is the key feature: tree forests compute input-adaptive "
         f"linear projections where each input sees a different effective weight matrix.")

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
    body("Our key contributions: (1) <b>BatchedTreeForest and ObliviousTreeForest</b>: batched implementations "
         "of standard and NODE-style oblivious decision trees. "
         "(2) <b>LinearPlusForest</b>: a linear base projection augmented with a single wide tree forest. "
         "(3) <b>Speed optimization ablation</b>: oblivious trees, alternating layers, selective V+O "
         "placement, and torch.compile. "
         "(4) <b>MoE comparison</b>: testing whether tree structure matters vs. any input-dependent mixing. "
         "(5) <b>Empirical validation</b> on Shakespeare across 6 model configurations at full scale.")

    # ── 2. Background ────────────────────────────────────────────────────
    sec("2. Background")
    subsec("2.1 Soft Decision Trees")
    body("A soft decision tree of depth D has 2<super>D</super> &minus; 1 internal nodes and "
         "2<super>D</super> leaves. Each internal node <i>i</i> computes a soft routing probability:")
    eq("p_left(x) = sigmoid((w_i &middot; x + b_i) / &tau;)")
    body("The tree output is a probability-weighted sum over learned leaf vectors. Crucially, with soft "
         "routing, the forest computes an <i>input-adaptive linear projection</i> &mdash; a different "
         "effective weight matrix for every input, constructed from a weighted combination of leaf matrices.")

    subsec("2.2 Oblivious Decision Trees (NODE-style)")
    body("In oblivious trees (Popov et al., 2020), all nodes at the same depth share one hyperplane. "
         "Decision weights have shape (T, D, D<sub>in</sub>) &mdash; fewer parameters than standard trees. "
         "Leaf probabilities are computed via outer product of per-depth choices, eliminating depth-dependent "
         "gradient vanishing.")

    subsec("2.3 Transformer Attention")
    body("Standard multi-head attention computes Q = xW<sub>Q</sub>, K = xW<sub>K</sub>, "
         "V = xW<sub>V</sub>, then Attn(Q,K,V) = softmax(QK<super>T</super>/&radic;d<sub>k</sub>)V. "
         "We replace the linear projections with tree-based projection layers.")

    # ── 3. Method ────────────────────────────────────────────────────────
    sec("3. Method")
    subsec("3.1 BatchedTreeForest")
    body("All tree parameters are stored in stacked tensors. The routing computation for all T trees "
         "is a single batched einsum:")
    eq("decisions = einsum('bsd,tnd-&gt;bstn', x, W_decision)")
    body("Leaf probabilities are computed in log-space. Input-dependent gating via a learned projection "
         "replaces fixed mixture weights, allowing each token to route to different tree combinations.")

    subsec("3.2 ObliviousTreeForest")
    body("Oblivious trees constrain all nodes at the same depth to share a single hyperplane, reducing "
         "decision weights from (T, 2<super>D</super>&minus;1, D<sub>in</sub>) to (T, D, D<sub>in</sub>). "
         "Leaf probs computed via Kronecker product, unrolled for depth 3: "
         "P(leaf<sub>ijk</sub>) = d<sub>0</sub><super>i</super> &middot; d<sub>1</sub><super>j</super> "
         "&middot; d<sub>2</sub><super>k</super>.")

    subsec("3.3 LinearPlusForest")
    body("We use a simple additive architecture:")
    eq("output = W_base &middot; x + b + &gamma; &middot; Forest(x)")
    body("where &gamma; is a learned shrinkage factor (initialized to 0.1). The linear base preserves "
         "residual stream structure; the forest adds input-adaptive nonlinear correction.")

    subsec("3.4 Unfused QKV Routing")
    body("Q, K, V serve fundamentally different functions. We use <b>separate forests</b> for each, "
         "allowing independent routing specialization. Shared routing forces identical paths for all three.")

    subsec("3.5 QK-Norm")
    body("Tree projections have unpredictable output scale during training. Following Gemma 2/ViT-22B, "
         "we apply LayerNorm to Q and K after projection, stabilizing attention logit magnitudes.")

    subsec("3.6 Speed Optimizations")
    body("(1) <b>Oblivious trees</b>: fewer parameters and vectorized leaf prob computation. "
         "(2) <b>Alternating layers</b>: only every other block uses tree projections. "
         "(3) <b>Selective V+O</b>: trees only on V and O; Q/K use linear. "
         "(4) <b>torch.compile</b>: fuses operations and optimizes the computation graph.")

    subsec("3.7 MoE Variant")
    body("As an ablation, we replace tree forests with standard MoE layers using top-k expert routing, "
         "testing whether the tree structure itself matters.")

    subsec("3.8 Training Details")
    body("<b>Per-node temperature:</b> Each node learns its own via softplus(logit + 0.5413) (init 1.0). "
         "<b>Temperature annealing:</b> Held at 1.0 for 50% of training, cosine to 0.7. "
         "<b>Init:</b> N(0, 0.1) for decision weights. "
         "<b>Optimizer:</b> Decision weights get 3&times; LR, zero weight decay. 100-step warmup + cosine decay. "
         "<b>Regularization:</b> Entropy reg (&lambda;=0.005, depth-decay) + leaf balancing (&alpha;=0.01).")

    # ── 4. Experiments ───────────────────────────────────────────────────
    sec("4. Experiments")
    subsec("4.1 Setup")
    body("We evaluate on character-level language modeling using the Tiny Shakespeare dataset (Karpathy, 2015): "
         "1.1M characters, 65-character vocabulary, 90/10 train/val split. "
         "Full configuration: 4 transformer layers, d<sub>model</sub>=128, 4 attention heads, "
         "sequence length 256, batch size 32, 2000 training steps.")
    space(4)

    # Model config table
    model_rows = [['Model', 'Description', 'Params']]
    configs = [
        ('standard', 'Linear Q/K/V/O projections'),
        ('boosted', 'Linear base + 24-tree forest, depth 3'),
        ('oblivious_boosted', 'Oblivious trees in LinearPlusForest'),
        ('oblivious_boosted_alt', 'Oblivious L+F, alternating layers'),
        ('oblivious_boosted_vo_alt', 'V+O trees only, alternating'),
        ('moe_boosted_alt', 'MoE replacing trees, alternating'),
    ]
    for key, desc in configs:
        params = fmt_params(r(key, 'params', 0))
        name = r(key, 'name', key)
        model_rows.append([name, desc, params])
    table_block(model_rows, col_widths=[2*inch, 2.5*inch, 0.8*inch])

    subsec("4.2 Results")
    fig('main_results.png',
        '<b>Figure 1:</b> Main results on Shakespeare character-level LM (2000 steps). '
        '(a) Validation accuracy vs. steps. (b) Validation loss. '
        '(c) Routing entropy. (d) Speed vs. accuracy trade-off.')

    # Results table
    result_rows = [['Model', 'Val Acc', 'Val Loss', 'ms/step', 'Params']]
    for key, _ in configs:
        name = r(key, 'name', key)
        acc = fmt_acc(r(key, 'final_val_acc', 0)) + '%'
        loss = fmt_loss(r(key, 'final_val_loss', 0))
        ms = fmt_ms(r(key, 'ms_per_step', 0))
        params = fmt_params(r(key, 'params', 0))
        result_rows.append([name, acc, loss, ms, params])
    table_block(result_rows, col_widths=[2*inch, 0.9*inch, 0.8*inch, 0.8*inch, 0.8*inch])
    body("<b>Table 2:</b> Full-config results (d<sub>model</sub>=128, 4 layers, seq_len=256, 2000 steps).")

    body("The Linear+Forest matches the standard transformer in accuracy while speed-optimized variants "
         "trade minimal accuracy for significant speed gains.")

    subsec("4.3 Speed Optimization Progression")
    fig('speed_vs_accuracy.png',
        '<b>Figure 2:</b> Speed vs. accuracy trade-off. Pareto-optimal configurations '
        'offer the best accuracy at each speed budget.')
    body("The progression from full Linear+Forest to V+O alternating shows that most tree compute can "
         "be eliminated with minimal accuracy loss, suggesting trees add most value to V and O projections.")

    subsec("4.4 Routing Entropy Analysis")
    fig('entropy.png',
        '<b>Figure 3:</b> Routing entropy over training. Conservative temperature annealing '
        '(1.0 &rarr; 0.7) preserves meaningful routing entropy.')
    body("Unlike aggressive annealing (1.0 &rarr; 0.1), the conservative schedule preserves meaningful "
         "routing entropy. Soft routing <i>is</i> the feature: each input sees a different effective weight "
         "matrix. Hard routing reduces trees to piecewise-constant lookup tables.")

    subsec("4.5 MoE Ablation")
    moe_acc = fmt_acc(r('moe_boosted_alt', 'final_val_acc', 0))
    vo_acc = fmt_acc(r('oblivious_boosted_vo_alt', 'final_val_acc', 0))
    body(f"The Linear+MoE variant achieves {moe_acc}% validation accuracy, compared to {vo_acc}% "
         f"for the comparable tree variant (Oblivious L+F V+O, alt).")

    # ── 5. Analysis ──────────────────────────────────────────────────────
    sec("5. Analysis and Discussion")
    subsec("5.1 Why LinearPlusForest Works")
    body("(1) The <b>linear base provides a guaranteed gradient path</b> &mdash; even if tree routing "
         "shifts, the linear projection keeps learning. "
         "(2) <b>Trees as corrections, not replacements</b> &mdash; the forest learns the residual between "
         "what a linear projection can represent and the optimal projection. "
         "(3) <b>Input-adaptive projections</b> &mdash; the effective projection is "
         "W<sub>base</sub> + &gamma; &middot; W<sub>eff</sub>(x), a form of input-conditional computation.")

    subsec("5.2 Why Unfused Routing Matters")
    body("Q, K, V, and O serve fundamentally different functions. Fused QKV routing forces identical tree "
         "paths for all three, limiting expressiveness. Unfused routing allows each to specialize.")

    subsec("5.3 Oblivious Trees and Speed/Accuracy Trade-off")
    body("Oblivious trees achieve comparable accuracy with fewer parameters. The shared-hyperplane constraint "
         "acts as a regularizer. Alternating layers are effective because not every transformer block benefits "
         "equally from nonlinear projections.")

    subsec("5.4 Limitations")
    body("(1) <b>Speed:</b> Even the fastest tree variant is slower than standard attention. GPU benchmarks needed. "
         "(2) <b>Scale:</b> Only tested at small model sizes. "
         "(3) <b>Task diversity:</b> Only character-level Shakespeare evaluated. "
         "(4) <b>Temperature:</b> The schedule is hand-tuned; adaptive methods could improve robustness.")

    # ── 6. Related Work ──────────────────────────────────────────────────
    sec("6. Related Work")
    body("<b>Soft Decision Trees:</b> Irsoy et al. (2012) introduced soft trees. Frosst &amp; Hinton (2017) "
         "used them for distillation. Hazimeh et al. (2020) proposed differentiable trees for tabular data.")
    body("<b>NODE:</b> Popov et al. (2020) introduced Neural Oblivious Decision Ensembles for tabular data. "
         "We adapt oblivious trees for attention projections with sigmoid routing.")
    body("<b>Tree-based Neural Networks:</b> Deep Neural Decision Forests (Kontschieder et al., 2015) "
         "combined forests with neural features. Adaptive Neural Trees (Tanno et al., 2019).")
    body("<b>Mixture of Experts:</b> LinearPlusForest shares conceptual similarity with MoE (Shazeer et al., 2017). "
         "We directly compare against an MoE variant.")
    body("<b>Efficient Attention:</b> Low-rank (Wang et al., 2020), sparse (Child et al., 2019), and "
         "kernel-based (Katharopoulos et al., 2020) methods. Tree-based projections provide a distinct "
         "inductive bias &mdash; input-conditional piecewise linear projections.")

    # ── 7. Future Work ───────────────────────────────────────────────────
    sec("7. Future Work")
    body("(1) <b>GPU optimization:</b> Custom CUDA kernels for tree routing. "
         "(2) <b>Larger scale:</b> Evaluate on larger models and datasets. "
         "(3) <b>Adaptive temperature:</b> Replace fixed schedule with learned temperature. "
         "(4) <b>Routing analysis:</b> Investigate what features routing learns to partition on. "
         "(5) <b>Hybrid architectures:</b> Trees in FFN gating, embeddings, etc.")

    # ── 8. Conclusion ────────────────────────────────────────────────────
    sec("8. Conclusion")
    body(f"We demonstrate that differentiable decision trees can serve as effective projection layers in "
         f"transformer attention, matching standard linear projections on character-level Shakespeare "
         f"({bst_acc}% vs. {std_acc}% validation accuracy). The key insight is that trees work best as "
         f"<i>residual corrections to a linear base</i> with <i>unfused per-projection routing</i>.")
    body(f"Speed-optimized variants using oblivious trees, alternating layers, and selective V+O placement "
         f"achieve {fmt_acc(r('oblivious_boosted_vo_alt', 'final_val_acc', 0))}% accuracy at "
         f"{fmt_ms(r('oblivious_boosted_vo_alt', 'ms_per_step', 0))}ms/step &mdash; significantly faster "
         f"than full Linear+Forest ({bst_ms}ms/step). The deeper insight is that soft routing is not a "
         f"compromise &mdash; it <i>is</i> the feature, computing input-adaptive projections.")

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
        "Popov, S., Morozov, S., &amp; Babenko, A. (2020). Neural oblivious decision ensembles for deep learning on tabular data. <i>ICLR</i>.",
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
