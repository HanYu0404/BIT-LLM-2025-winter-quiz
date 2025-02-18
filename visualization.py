import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 加载embedding数据
embeddings = np.loadtxt("embeddings.txt")

# t-SNE降维
tsne = TSNE(n_components=2,
            perplexity=30,
            n_iter=5000,
            random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# 可视化
plt.figure(figsize=(12, 8))
plt.scatter(embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            s=30,
            alpha=0.6,
            c='royalblue',
            edgecolor='w')

plt.title('t-SNE Visualization of Plasmid Embeddings', fontsize=14)
plt.xlabel('t-SNE Dimension 1', fontsize=12)
plt.ylabel('t-SNE Dimension 2', fontsize=12)
plt.grid(alpha=0.3)
plt.gca().set_facecolor('whitesmoke')

plt.savefig('tsne_visualization.png', dpi=300, bbox_inches='tight')
plt.show()