import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def make_charts():
    scores_file = os.path.join("data", "results", "scores.csv")
    if not os.path.exists(scores_file):
        print("scores.csv not found")
        return
        
    df = pd.read_csv(scores_file)
    os.makedirs("figures", exist_ok=True)
    
    # Calculate derived metrics
    df['hallucination_binary'] = df['hallucination_flag'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
    df['error_binary'] = df['error_flag'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
    
    sns.set_theme(style="darkgrid")
    
    # CHART 1: Hallucination reduction
    plt.figure(figsize=(10, 6))
    hal_rate = df.groupby(['model', 'type'])['hallucination_binary'].mean() * 100
    hal_rate = hal_rate.reset_index()
    sns.barplot(data=hal_rate, x='model', y='hallucination_binary', hue='type', palette='Set2')
    plt.title('Hallucination Rate: Ungrounded vs Grounded')
    plt.ylabel('Hallucination Rate (%)')
    plt.savefig(os.path.join("figures", "hallucination_reduction.png"), dpi=300)
    plt.close()
    
    # CHART 2: Error rate comparison
    plt.figure(figsize=(10, 6))
    err_rate = df.groupby(['model', 'type'])['error_binary'].mean() * 100
    err_rate = err_rate.reset_index()
    sns.barplot(data=err_rate, x='model', y='error_binary', hue='type', palette='Set1')
    plt.title('Error Rate: Ungrounded vs Grounded')
    plt.ylabel('Error Rate (%)')
    plt.savefig(os.path.join("figures", "error_comparison.png"), dpi=300)
    plt.close()
    
    # CHART 3: Score distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='total_score', hue='type', multiple='dodge', bins=8, palette='Set3')
    plt.title('Score Distribution: Ungrounded vs Grounded')
    plt.xlabel('Total Score')
    plt.savefig(os.path.join("figures", "score_distribution.png"), dpi=300)
    plt.close()
    
    # CHART 4: Category-level performance
    plt.figure(figsize=(12, 6))
    cat_err = df.groupby(['category', 'type'])['error_binary'].mean() * 100
    cat_err = cat_err.reset_index()
    sns.barplot(data=cat_err, x='category', y='error_binary', hue='type', palette='Paired')
    plt.title('Error Rate by Category')
    plt.ylabel('Error Rate (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join("figures", "error_by_category.png"), dpi=300)
    plt.close()
    
    # PRINT METRICS
    overall_hal_un = df[df['type'] == 'ungrounded']['hallucination_binary'].mean() * 100
    overall_hal_gr = df[df['type'] == 'grounded']['hallucination_binary'].mean() * 100
    
    overall_err_un = df[df['type'] == 'ungrounded']['error_binary'].mean() * 100
    overall_err_gr = df[df['type'] == 'grounded']['error_binary'].mean() * 100
    
    print("\n=== PHARMRAG RESULTS ===")
    print("\nHallucination reduction:")
    print(f"Ungrounded: {overall_hal_un:.1f}%")
    print(f"Grounded: {overall_hal_gr:.1f}%")
    
    print("\nError reduction:")
    print(f"Ungrounded: {overall_err_un:.1f}%")
    print(f"Grounded: {overall_err_gr:.1f}%")
    
    print("\nPer model stats:")
    for model in df['model'].unique():
        m_df = df[df['model'] == model]
        h_un = m_df[m_df['type'] == 'ungrounded']['hallucination_binary'].mean() * 100
        h_gr = m_df[m_df['type'] == 'grounded']['hallucination_binary'].mean() * 100
        print(f"\n{model}:")
        print(f"  Hallucination: {h_un:.1f}% -> {h_gr:.1f}%")

if __name__ == "__main__":
    make_charts()
