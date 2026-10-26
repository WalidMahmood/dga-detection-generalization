import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def create_architecture_diagram():
    fig, axes = plt.subplots(1, 2, figsize=(18, 11), dpi=300)
    fig.patch.set_facecolor('#ffffff')

    # Color palette
    c_bg_box = '#f8fafc'
    c_border = '#cbd5e1'
    c_char = '#2563eb'     # Blue
    c_word = '#059669'     # Green
    c_cohere = '#7c3aed'   # Purple
    c_gate = '#d97706'     # Amber
    c_head = '#dc2626'     # Red
    c_text = '#0f172a'
    c_sub = '#475569'

    for ax in axes:
        ax.set_facecolor('#ffffff')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 14)
        ax.axis('off')

    # ==========================================
    # LEFT: Dual-Branch Perfect (v3) [4.10M params]
    # ==========================================
    ax1 = axes[0]
    ax1.set_title("Architecture A: Dual-Branch Perfect (v3)\n[Baseline Dual-Branch • 4.10M Parameters]", 
                  fontsize=14, fontweight='bold', color='#1e293b', pad=20)

    # Input Box
    ax1.add_patch(patches.FancyBboxPatch((2.5, 12.2), 5.0, 1.0, boxstyle="round,pad=0.2", 
                                        ec=c_border, fc=c_bg_box, lw=1.5))
    ax1.text(5.0, 12.8, "Raw Domain SLD String", fontsize=11, fontweight='bold', ha='center', va='center', color=c_text)
    ax1.text(5.0, 12.4, "Example: 'sugarmouse' (length L)", fontsize=9, ha='center', va='center', color=c_sub, style='italic')

    # Arrow Down split
    ax1.annotate('', xy=(3.0, 10.8), xytext=(3.5, 12.0), arrowprops=dict(arrowstyle="->", lw=2, color=c_sub))
    ax1.annotate('', xy=(7.0, 10.8), xytext=(6.5, 12.0), arrowprops=dict(arrowstyle="->", lw=2, color=c_sub))

    # Branch 1: Character (Left)
    ax1.add_patch(patches.FancyBboxPatch((1.0, 9.2), 3.8, 1.4, boxstyle="round,pad=0.2", ec=c_char, fc='#eff6ff', lw=1.8))
    ax1.text(2.9, 10.2, "Character Branch (IDs)", fontsize=10, fontweight='bold', ha='center', va='center', color=c_char)
    ax1.text(2.9, 9.6, "One-Hot / Embedding (38 → 64)\nBiLSTM (64, hidden=128)", fontsize=8.5, ha='center', va='center', color=c_text)

    # Branch 2: Word (Right)
    ax1.add_patch(patches.FancyBboxPatch((5.2, 9.2), 3.8, 1.4, boxstyle="round,pad=0.2", ec=c_word, fc='#ecfdf5', lw=1.8))
    ax1.text(7.1, 10.2, "Word Sequence Branch", fontsize=10, fontweight='bold', ha='center', va='center', color=c_word)
    ax1.text(7.1, 9.6, "Viterbi Split (wordninja)\nFastText 300D → BiLSTM (300, 128)", fontsize=8.5, ha='center', va='center', color=c_text)

    # Vectors h_char and h_word
    ax1.annotate('', xy=(2.9, 7.8), xytext=(2.9, 9.0), arrowprops=dict(arrowstyle="->", lw=2, color=c_char))
    ax1.annotate('', xy=(7.1, 7.8), xytext=(7.1, 9.0), arrowprops=dict(arrowstyle="->", lw=2, color=c_word))
    
    ax1.add_patch(patches.Rectangle((1.7, 7.2), 2.4, 0.6, ec=c_char, fc='#dbeafe', lw=1.2, ls='--'))
    ax1.text(2.9, 7.5, r"$h_{char} \in \mathbb{R}^{256}$", fontsize=9, fontweight='bold', ha='center', va='center', color=c_char)

    ax1.add_patch(patches.Rectangle((5.9, 7.2), 2.4, 0.6, ec=c_word, fc='#d1fae5', lw=1.2, ls='--'))
    ax1.text(7.1, 7.5, r"$h_{word} \in \mathbb{R}^{256}$", fontsize=9, fontweight='bold', ha='center', va='center', color=c_word)

    # Gated Fusion
    ax1.annotate('', xy=(4.2, 5.8), xytext=(3.0, 7.1), arrowprops=dict(arrowstyle="->", lw=2, color=c_sub))
    ax1.annotate('', xy=(5.8, 5.8), xytext=(7.0, 7.1), arrowprops=dict(arrowstyle="->", lw=2, color=c_sub))

    ax1.add_patch(patches.FancyBboxPatch((3.0, 4.6), 4.0, 1.2, boxstyle="round,pad=0.2", ec=c_gate, fc='#fffbeb', lw=1.8))
    ax1.text(5.0, 5.4, "Learnable Gated Fusion", fontsize=10, fontweight='bold', ha='center', va='center', color=c_gate)
    ax1.text(5.0, 4.9, r"$g = \sigma(W_g [h_{char} \,\|\, h_{word}] + b_g)$" + "\n" + r"$h_{fused} = g \odot h_{char} + (1-g) \odot h_{word}$", fontsize=8, ha='center', va='center', color=c_text)

    # Arrow to Classifier Head
    ax1.annotate('', xy=(5.0, 3.2), xytext=(5.0, 4.4), arrowprops=dict(arrowstyle="->", lw=2, color=c_sub))
    
    # Classification Head (Heavy MLP)
    ax1.add_patch(patches.FancyBboxPatch((2.2, 1.8), 5.6, 1.3, boxstyle="round,pad=0.2", ec=c_head, fc='#fef2f2', lw=1.8))
    ax1.text(5.0, 2.7, "Wide MLP Classification Head (4.10M params)", fontsize=10, fontweight='bold', ha='center', va='center', color=c_head)
    ax1.text(5.0, 2.1, "Dense(256 → 1024 → 512 → 1) • Dropout(0.4)\nCross-Entropy Loss (Standard)", fontsize=8, ha='center', va='center', color=c_text)

    # Output
    ax1.annotate('', xy=(5.0, 0.7), xytext=(5.0, 1.7), arrowprops=dict(arrowstyle="->", lw=2, color=c_head))
    ax1.add_patch(patches.Circle((5.0, 0.4), 0.45, ec=c_head, fc='#fee2e2', lw=1.8))
    ax1.text(5.0, 0.4, r"$\hat{y}$", fontsize=11, fontweight='bold', ha='center', va='center', color=c_head)
    ax1.text(5.0, -0.2, "P(DGA)", fontsize=9, fontweight='bold', ha='center', va='center', color=c_sub)

    # Limitation note on A
    ax1.text(5.0, 6.4, "[Limitation] No Explicit Semantic Similarity Check\n(Relies solely on sequential letter patterns)", 
             fontsize=8, ha='center', va='center', color='#991b1b', bbox=dict(boxstyle="round,pad=0.3", fc="#fef2f2", ec="#fca5a5"))


    # ==========================================================
    # RIGHT: Dual-Branch Coherent Architecture [0.73M params]
    # ==========================================================
    ax2 = axes[1]
    ax2.set_title("Architecture B: Dual-Branch Coherent (Proposed)\n[Semantic-Augmented • 0.73M Parameters (204× Smaller)]", 
                  fontsize=14, fontweight='bold', color='#1e293b', pad=20)

    # Input Box
    ax2.add_patch(patches.FancyBboxPatch((2.5, 12.2), 5.0, 1.0, boxstyle="round,pad=0.2", ec=c_border, fc=c_bg_box, lw=1.5))
    ax2.text(5.0, 12.8, "Raw Domain SLD String", fontsize=11, fontweight='bold', ha='center', va='center', color=c_text)
    ax2.text(5.0, 12.4, "Example: 'sugarmouse' (length L)", fontsize=9, ha='center', va='center', color=c_sub, style='italic')

    # 3-Way Split Arrows
    ax2.annotate('', xy=(2.0, 10.8), xytext=(3.5, 12.0), arrowprops=dict(arrowstyle="->", lw=2, color=c_sub))
    ax2.annotate('', xy=(5.0, 10.8), xytext=(5.0, 12.0), arrowprops=dict(arrowstyle="->", lw=2, color=c_sub))
    ax2.annotate('', xy=(8.0, 10.8), xytext=(6.5, 12.0), arrowprops=dict(arrowstyle="->", lw=2, color=c_sub))

    # Branch 1: Char
    ax2.add_patch(patches.FancyBboxPatch((0.5, 9.2), 2.8, 1.4, boxstyle="round,pad=0.2", ec=c_char, fc='#eff6ff', lw=1.8))
    ax2.text(1.9, 10.2, "Character Branch", fontsize=9.5, fontweight='bold', ha='center', va='center', color=c_char)
    ax2.text(1.9, 9.6, "Embedding(38, 64)\nBiLSTM(64, 128)\n+ Char Dropout(0.05)", fontsize=7.5, ha='center', va='center', color=c_text)

    # Branch 2: Word
    ax2.add_patch(patches.FancyBboxPatch((3.6, 9.2), 2.8, 1.4, boxstyle="round,pad=0.2", ec=c_word, fc='#ecfdf5', lw=1.8))
    ax2.text(5.0, 10.2, "Word Branch", fontsize=9.5, fontweight='bold', ha='center', va='center', color=c_word)
    ax2.text(5.0, 9.6, "Viterbi (wordninja)\nFastText 300D\nBiLSTM(300, 128)", fontsize=7.5, ha='center', va='center', color=c_text)

    # Branch 3: Explicit 6D Semantic Coherence Vector
    ax2.add_patch(patches.FancyBboxPatch((6.7, 8.8), 3.0, 1.8, boxstyle="round,pad=0.2", ec=c_cohere, fc='#f5f3ff', lw=2.0))
    ax2.text(8.2, 10.2, "6D Semantic Coherence", fontsize=9.5, fontweight='bold', ha='center', va='center', color=c_cohere)
    cohere_desc = "Pairwise Cosine Sim S:\n• v1: min(S) [clash]\n• v2: mean(S) [coherence]\n• v3: max(S) [peak]\n• v4: Token Count k/6\n• v5: Dict Coverage\n• v6: Hyphen Flag"
    ax2.text(8.2, 9.4, cohere_desc, fontsize=7.0, ha='center', va='center', color=c_text)

    # Vectors
    ax2.annotate('', xy=(1.9, 7.8), xytext=(1.9, 9.0), arrowprops=dict(arrowstyle="->", lw=2, color=c_char))
    ax2.annotate('', xy=(5.0, 7.8), xytext=(5.0, 9.0), arrowprops=dict(arrowstyle="->", lw=2, color=c_word))
    
    ax2.add_patch(patches.Rectangle((0.9, 7.2), 2.0, 0.6, ec=c_char, fc='#dbeafe', lw=1.2, ls='--'))
    ax2.text(1.9, 7.5, r"$h_{char} \in \mathbb{R}^{256}$", fontsize=8.5, fontweight='bold', ha='center', va='center', color=c_char)

    ax2.add_patch(patches.Rectangle((4.0, 7.2), 2.0, 0.6, ec=c_word, fc='#d1fae5', lw=1.2, ls='--'))
    ax2.text(5.0, 7.5, r"$h_{word} \in \mathbb{R}^{256}$", fontsize=8.5, fontweight='bold', ha='center', va='center', color=c_word)

    # Gated Fusion of Char + Word
    ax2.annotate('', xy=(3.0, 5.8), xytext=(2.2, 7.1), arrowprops=dict(arrowstyle="->", lw=2, color=c_sub))
    ax2.annotate('', xy=(4.0, 5.8), xytext=(4.8, 7.1), arrowprops=dict(arrowstyle="->", lw=2, color=c_sub))

    ax2.add_patch(patches.FancyBboxPatch((1.6, 4.6), 3.8, 1.2, boxstyle="round,pad=0.2", ec=c_gate, fc='#fffbeb', lw=1.8))
    ax2.text(3.5, 5.4, "Learnable Gated Fusion", fontsize=9.5, fontweight='bold', ha='center', va='center', color=c_gate)
    ax2.text(3.5, 4.9, r"$h_{fused} = g \odot h_{char} + (1-g) \odot h_{word}$" + "\n" + r"$g = \sigma(W_g [h_{char} \,\|\, h_{word}] + b_g)$", fontsize=7.5, ha='center', va='center', color=c_text)

    # Concatenation with 6D Coherence Vector
    ax2.annotate('', xy=(4.5, 3.4), xytext=(3.5, 4.4), arrowprops=dict(arrowstyle="->", lw=2, color=c_gate))
    ax2.annotate('', xy=(5.5, 3.4), xytext=(8.0, 8.6), arrowprops=dict(arrowstyle="->", lw=2, color=c_cohere, connectionstyle="angle,angleA=-90,angleB=180,rad=10"))

    ax2.add_patch(patches.FancyBboxPatch((2.2, 2.7), 5.6, 0.7, boxstyle="round,pad=0.2", ec='#0284c7', fc='#f0f9ff', lw=1.5))
    ax2.text(5.0, 3.05, r"Feature Concatenation: $h_{final} = [h_{fused} \,\|\, v_{coherence}] \in \mathbb{R}^{262}$", fontsize=8.5, fontweight='bold', ha='center', va='center', color='#0369a1')

    # Compact MLP Head
    ax2.annotate('', xy=(5.0, 2.2), xytext=(5.0, 2.6), arrowprops=dict(arrowstyle="->", lw=2, color=c_sub))
    ax2.add_patch(patches.FancyBboxPatch((2.0, 1.2), 6.0, 1.0, boxstyle="round,pad=0.2", ec=c_head, fc='#fef2f2', lw=1.8))
    ax2.text(5.0, 1.8, "Compact MLP Head (0.73M Total Parameters)", fontsize=9.5, fontweight='bold', ha='center', va='center', color=c_head)
    ax2.text(5.0, 1.4, "Dense(262 → 128 → 1) • Label Smoothing (0.05) • Dropout (0.4)\nCheckpoint Criterion: Highest Macro-Recall on Validation Wordlists", fontsize=7.5, ha='center', va='center', color=c_text)

    # Output
    ax2.annotate('', xy=(5.0, 0.6), xytext=(5.0, 1.1), arrowprops=dict(arrowstyle="->", lw=2, color=c_head))
    ax2.add_patch(patches.Circle((5.0, 0.35), 0.38, ec=c_head, fc='#fee2e2', lw=1.8))
    ax2.text(5.0, 0.35, r"$\hat{y}$", fontsize=10.5, fontweight='bold', ha='center', va='center', color=c_head)
    ax2.text(5.0, -0.15, "P(DGA)", fontsize=8.5, fontweight='bold', ha='center', va='center', color=c_sub)

    # Victory badge on B
    ax2.text(8.2, 5.0, "* OUTPERFORMS ModernBERT on Matsnu *\n(17.60% vs 14.15% • +24.4% Relative)\n204x Fewer Parameters (~3MB)", 
             fontsize=8.5, ha='center', va='center', color='#166534', fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.4", fc="#dcfce7", ec="#86efac", lw=1.5))

    plt.tight_layout()
    os.makedirs('reports/figures', exist_ok=True)
    save_path = 'reports/figures/model_architectures_comparison.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Diagram successfully saved to {save_path}")

if __name__ == '__main__':
    create_architecture_diagram()
