import matplotlib.pyplot as plt

# Data akurasi skenario
skenario_labels = [
    'Single 1 (97%)', 'Single 2 (78%)', 'Single 3 (77%)',
    'Multi 1 (96%)', 'Multi 2 (71%)', 'Multi 3 (68%)', 'Multi 4 (89%)'
]
akurasi = [97, 78, 77, 96, 71, 68, 89]

# Warna batang untuk pembeda single dan multi
colors = ['green'] * 3 + ['yellow'] * 4

# Membuat bar chart
plt.figure(figsize=(12, 6))
bars = plt.bar(skenario_labels, akurasi, color=colors)

# Menambahkan nilai akurasi di atas batang
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height}%', ha='center', va='bottom', fontsize=10)

# Judul dan label
plt.title('Perbandingan Akurasi Deteksi APD per Skenario', fontsize=14)
plt.xlabel('Skenario Pengujian')
plt.ylabel('Akurasi (%)')
plt.ylim(0, 110)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Menampilkan grafik
plt.show()