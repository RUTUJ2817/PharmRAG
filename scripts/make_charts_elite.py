import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def make_elite_charts():
    scores_file = os.path.join("data", "results", "scores_elite.csv")
    if not os.path.exists(scores_file):
        return
        
    df = pd.read_csv(scores_file)
    os.makedirs("figures", exist_ok=True)
    sns.set_theme(style="darkgrid")
    
    # Chart 1: Safety Score across levels
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='level', y='Safety_Score', hue='type', palette='magma')
    plt.title('Safety Score by Query Difficulty Level (Hybrid RAG Ablation)')
    plt.ylabel('Safety Score (0-10)')
    plt.xlabel('Query Level (1=Factual, 2=Multi-hop, 3=Adversarial)')
    plt.savefig(os.path.join("figures", "elite_safety_ablation.png"), dpi=300)
    plt.close()
    
    # Chart 2: Taxonomy Error Distribution
    errors_df = df[df['Taxonomy'] != 'None']
    if not errors_df.empty:
        plt.figure(figsize=(10, 6))
        sns.countplot(data=errors_df, x='Taxonomy', hue='type', palette='Set2')
        plt.title('Error Taxonomy Distribution')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join("figures", "elite_error_taxonomy.png"), dpi=300)
        plt.close()
    
    # Metrics Printout
    print("\n=== PHARMRAG ELITE RESULTS ===")
    
    # Safety score overall
    s_un = df[df['type'] == 'ungrounded']['Safety_Score'].mean()
    s_gr = df[df['type'] == 'grounded']['Safety_Score'].mean()
    print(f"\nOverall Safety Score (out of 10):")
    print(f"Ungrounded: {s_un:.1f}")
    print(f"Hybrid Grounded: {s_gr:.1f}")
    
    # NLI Entailment
    entailment_pct = (df[df['Factual_Consistency'] == 'Entailment'].shape[0] / df[df['type'] == 'grounded'].shape[0]) * 100
    print(f"\nFactual Consistency (NLI Entailment Rate): {entailment_pct:.1f}%")

if __name__ == "__main__":
    make_elite_charts()
