# Predictive Analytics - Crop Recommendation
Domain yang saya pilih untuk proyek machine learning ini adalah Pertanian, dengan judul Rekomendasi jenis tanaman berdasarkan kondisi lingkungan.

## 1. Domain Proyek
### Latar Belakang:
Indonesia sebagai negara agraris memiliki tantangan dalam pemilihan jenis tanaman yang sesuai dengan kondisi lingkungan (nutrisi tanah, suhu, kelembapan, pH, curah hujan). Salah memilih jenis tanaman dapat menyebabkan gagal panen dan kerugian ekonomi.
### Masalah:
Kebanyakan petani masih mengandalkan intuisi dan pengalaman tanpa dasar data yang kuat. Diperlukan sistem berbasis machine learning untuk membantu rekomendasi jenis tanaman yang cocok berdasarkan data tanah dan cuaca.
### Referensi:
Dataset berasal dari Kaggle:
Crop Recommendation Dataset by Atharva Ingle
(https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)

Dataset ini digunakan dalam banyak penelitian, contohnya:
- “Crop Recommendation System using Machine Learning” (IJSRCS, 2018)
- “Agricultural Yield Prediction using Supervised Learning” (IEEE, 2020)

## 2. Business Understanding
### Problem Statements:
- Bagaimana cara memprediksi jenis tanaman yang cocok ditanam berdasarkan kondisi tanah dan cuaca?

### Goals:
- Membangun model klasifikasi yang dapat merekomendasikan jenis tanaman berdasarkan input numerik seperti N, P, K, suhu, kelembapan, pH, dan curah hujan.

### Solution Statement:
Untuk mencapai tujuan proyek yaitu membangun sistem rekomendasi tanaman berbasis data tanah dan cuaca, dilakukan beberapa pendekatan dan langkah solusi sebagai berikut:
1. Memahami Data
Langkah awal dilakukan dengan memahami data melalui analisis univariat dan multivariat, termasuk eksplorasi visualisasi seperti histogram, countplot, dan heatmap korelasi. Proses ini membantu dalam memahami distribusi data, korelasi antar fitur numerik (seperti nitrogen, fosfor, kalium, suhu, kelembapan, pH, dan curah hujan), serta mendeteksi ketidakseimbangan kelas dan potensi outlier.
2. Pembersihan dan Persiapan Data
Data dipersiapkan dengan langkah-langkah sebagai berikut:
- Label encoding pada variabel target (label tanaman) untuk digunakan dalam klasifikasi.
- Split data ke dalam data latih dan data uji dengan stratifikasi agar proporsi kelas tetap terjaga.
3. Pembuatan dan Evaluasi Model
- Random Forest adalah algoritma ensemble berbasis decision tree yang membangun banyak pohon keputusan dan menggabungkan hasilnya. Algoritma ini dikenal kuat terhadap overfitting dan dapat menangani data dengan baik meskipun tanpa normalisasi. Model ini digunakan sebagai baseline karena kestabilannya dan interpretasi fiturnya yang mudah dipahami.
- XGBoost (Extreme Gradient Boosting) adalah metode boosting yang sangat efisien dan akurat. Algoritma ini membangun model secara berurutan, di mana setiap model baru mencoba memperbaiki kesalahan dari model sebelumnya. XGBoost digunakan untuk meningkatkan akurasi model karena kemampuannya menangkap pola kompleks dan keunggulannya dalam banyak kompetisi data science.
- SVM adalah algoritma klasifikasi berbasis margin yang bekerja dengan mencari hyperplane terbaik yang memisahkan kelas-kelas data. Model ini efektif pada data berdimensi tinggi dan cocok digunakan pada dataset berskala kecil-menengah. Untuk meningkatkan performanya, fitur dinormalisasi agar skala data seragam.

## 3. Data Understanding
### Jumlah Data:
- 2.200 baris, 7 fitur input + 1 label target
- Tidak ada missing value
- Label terdiri dari 22 jenis tanaman, distribusi label seimbang
![alt text](https://github.com/AtikaOktavianti/-Predictive-Analytics/blob/main/isi%20dataset.png?raw=true)

Gambar 1. Isi Dataset Crop Recommendation
### Fitur:
| Fitur | Deskripsi |
| ------ | ------ |
| N |Kandungan Nitrogen |
| P |Kandungan Fosfor |
| K | Kandungan Kalium |
| temperature | Suhu Udara (°C) |
| humidity | Kelembapan udara (%) |
| ph | Keasaman tanah |
| rainfall | Curah hujan (mm) |
| label | Target (nama tanaman) |

Tabel 1. Fitur Dataset
### Visualisasi:
- Distribusi tanaman divisualisasikan dengan countplot
![alt text](https://github.com/AtikaOktavianti/-Predictive-Analytics/blob/main/distribusi.png?raw=true)
Gambar 2. Distribusi Tanaman pada dataset

Distribusi tanaman yang seimbang adalah kondisi ideal untuk pelatihan model klasifikasi, karena meningkatkan keadilan prediksi dan menyederhanakan proses evaluasi dan pemodelan.

- Korelasi antar fitur divisualisasikan dengan heatmap
![alt text](https://github.com/AtikaOktavianti/-Predictive-Analytics/blob/main/korelasi%20antar%20fitur.png?raw=true)
Gambar 3. Korelasi antar fitur

Hampir semua fitur tidak memiliki hubungan korelasi yang kuat satu sama lain, kecuali antara P dan K. Ini menunjukkan bahwa sebagian besar fitur dapat dianggap independen dan tidak redundant, baik untuk analisis maupun pemodelan machine learning.

contoh kode:
```
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='label', order=df['label'].value_counts().index)
plt.xticks(rotation=90)
```
## 4. Data Preparation
Proses yang dilakukan:
1. Label Encoding pada Target (Kolom label)
Label (nama tanaman) berupa data kategorikal bertipe string (seperti “rice”, “wheat”, “mungbean”, dll). Karena model machine learning hanya bisa memproses data numerik, maka label tersebut dikonversi ke bentuk angka dengan menggunakan LabelEncoder dari scikit-learn.
Misalnya: 'rice' → 0, 'maize' → 1, dst.
2. Pembagian Data (Train-Test Split)
Dataset dibagi menjadi dua bagian: 80% data untuk pelatihan (training set) dan 20% untuk pengujian (test set).
Proses pembagian dilakukan secara stratified (StratifiedSplit) untuk menjaga proporsi kelas target tetap sama pada data train dan test. Ini penting agar evaluasi model tidak bias terhadap distribusi kelas.

## 5. Modelling
Dalam tahap ini, dilakukan pembangunan dan pelatihan model machine learning dengan tiga algoritma berbeda, yaitu: Random Forest, XGBoost, dan Support Vector Machine (SVM). Ketiga model ini dipilih karena karakteristiknya yang cocok untuk klasifikasi multikelas, serta performa yang terbukti kuat pada berbagai tugas klasifikasi data terstruktur.
### Random Forest
Random Forest merupakan algoritma berbasis ensemble learning yang terdiri dari banyak decision tree. Setiap pohon dilatih dengan subset acak dari data (bootstrap) dan menggunakan subset acak dari fitur pada setiap pemisahan node. Hasil prediksi dari setiap pohon kemudian digabungkan (voting mayoritas) untuk menghasilkan keputusan akhir.
1. Parameter:
- Menggunakan default parameter dari Scikit-Learn
- random_state = 42 untuk replikasi hasil (kontrol randomisasi).
2. Karakteristik:
- Cepat dan andal, cocok untuk data tabular dengan fitur numerik.
- Tidak memerlukan scaling karena pemisahan dilakukan berdasarkan nilai ambang (threshold), bukan jarak antar titik.
- Tahan terhadap outlier dan multikolinearitas antar fitur.
- Dapat memberikan estimasi pentingnya fitur (feature importance).
3. Kelebihan:
- Robust terhadap overfitting jika jumlah tree cukup besar.
- Mudah diinterpretasikan dan digunakan sebagai baseline yang kuat.
4. Kekurangan:
- Kurang efisien pada data yang sangat besar (karena banyak pohon yang harus dievaluasi).
- Interpretasi tidak semudah model linear/logistik pada level matematis.
![alt text](https://github.com/AtikaOktavianti/-Predictive-Analytics/blob/main/CM%20RF.png?raw=true)
Gambar 4. Confussion Matrix Random forest

### XGBoost
XGBoost (Extreme Gradient Boosting) adalah metode boosting berbasis pohon yang membangun model secara sekuensial untuk memperbaiki kesalahan dari model sebelumnya. Algoritma ini mengoptimalkan fungsi objektif menggunakan metode gradient descent dan dilengkapi dengan regularisasi L1 dan L2 untuk mencegah overfitting.
1. Parameter:
- Booster: 'gbtree' (default)
- Tidak dilakukan hyperparameter tuning pada eksperimen ini.
- Model menggunakan parameter default karena sudah sangat baik untuk baseline.
2. Karakteristik:
- Sangat powerful dan sering digunakan di kompetisi data science (misalnya Kaggle).
- Menggabungkan kekuatan ensemble dan regularisasi.
- Mampu menangkap hubungan non-linear dan interaksi fitur kompleks.
3. Kelebihan
- Performa sangat baik pada dataset tabular.
- Mendukung early stopping dan parallel training.
- Mendukung missing value secara internal.
4. Kekurangan:
- Waktu pelatihan lebih lama dibanding Random Forest.
- Lebih kompleks untuk dituning.
![alt text](https://github.com/AtikaOktavianti/-Predictive-Analytics/blob/main/CM%20XGB.png?raw=true)
Gambar 5. Confussion Matrix XGBoost

### Support Vector Machine (SVM)
Support Vector Machine adalah algoritma klasifikasi yang bekerja dengan mencari hyperplane terbaik yang memisahkan kelas-kelas data dengan margin maksimum. SVM sangat efektif untuk dataset berdimensi tinggi dan dapat bekerja dengan kernel non-linear (RBF, polynomial, dll).
1. Parameter:
- Kernel = 'linear' (SVC dengan linear decision boundary)
2. Karakteristik:
- Dapat bekerja dengan baik pada dataset kecil hingga menengah.
- Dikenal stabil dalam kasus data yang tidak terlalu kompleks.
- Memiliki teori matematis yang kuat dan hasil prediksi yang presisi jika parameter dan preprocessing tepat.
3. Kelebihan:
- Akurat pada data bersih dan terstruktur.
- Mampu mengatasi kasus klasifikasi dengan margin tipis.
4. Kekurangan:
- Tidak scalable untuk dataset besar karena kompleksitas komputasi O(n²) atau lebih.
- Membutuhkan tuning kernel dan regularisasi agar performa optimal.
- Sensitif terhadap outlier dan skala fitur, sehingga wajib melakukan normalisasi.
![alt text](https://github.com/AtikaOktavianti/-Predictive-Analytics/blob/main/CM%20SVM.png?raw=true)
Gambar 6. Confussion Matrix SVM

## 6. Evaluasi
### Metrik yang digunakan:
1. Accuracy = (TP + TN) / (Total)
2. Precision = TP / (TP + FP)
3. Recall = TP / (TP + FN)

Penjelasan:

- TP (True Positive): Jumlah data positif yang diprediksi dengan benar sebagai positif.
- TN (True Negative): Jumlah data negatif yang diprediksi dengan benar - sebagai negatif.
- FP (False Positive): Jumlah data negatif yang diprediksi secara tidak benar sebagai positif (Kesalahan Tipe I).
- FN (False Negative): Jumlah data positif yang diprediksi secara tidak benar sebagai negatif (Kesalahan Tipe II).

### 1. Hasil accuracy 3 model yang dilatih:
| Model | Accuracy |
| ------ | ------ |
| Random Forest | 0.99 |
| XGBoost | 0.99 |
| SVM | 0.98 |

Tabel 2. Accuracy model

### 2. Precision, Recall, dan F1-score
**a. Random Forest**

Macro avg Precision / Recall / F1: 1.00 / 1.00 / 1.00

Hampir semua kelas diprediksi sempurna, dengan beberapa pengecualian seperti label 2 (f1-score 0.97) dan 20 (f1-score 0.97) yang menunjukkan ada sedikit ketidaksempurnaan. Selanjutnya, tidak ada kelas yang sangat lemah; model sangat stabil dan andal.

![alt text](https://github.com/AtikaOktavianti/-Predictive-Analytics/blob/main/RF.png?raw=true)

Gambar 7. Hasil Evaluasi Random Forest

**b. XGBoost**

Macro avg Precision / Recall / F1: 0.99 / 0.99 / 0.99

Beberapa kelas menunjukkan penurunan kecil, misalnya label 10 (recall 0.90, f1-score 0.95), 13, dan 14 (f1-score 0.98). Meskipun akurasi hampir sama dengan Random Forest, XGBoost sedikit kurang konsisten di beberapa kelas.

![alt text](https://github.com/AtikaOktavianti/-Predictive-Analytics/blob/main/XGB.png?raw=true)

Gambar 8. Hasil Evaluasi XGBoost

**c. Support Vector Machine (SVM)**

Macro avg Precision / Recall / F1: 0.99 / 0.98 / 0.98

Performa sedikit lebih rendah terutama pada kelas rice (recall 0.75, f1-score 0.86) dan jute (precision 0.80, f1-score 0.89), yang menandakan kemungkinan kesulitan model mengenali ciri khas dari kelas-kelas ini. Kelas lain sebagian besar diprediksi dengan baik (f1-score ≥ 0.97).

![alt text](https://github.com/AtikaOktavianti/-Predictive-Analytics/blob/main/svm.png?raw=true)

Gambar 9. Hasil Evaluasi SVM

### Analisis Performa Antar Kelas
1. Random Forest adalah model yang paling stabil dengan distribusi metrik yang hampir sempurna di semua kelas.
2. XGBoost menunjukkan performa mendekati Random Forest, tetapi memiliki ketidaksempurnaan kecil pada beberapa label, terutama pada recall.
3. SVM cenderung memiliki performa fluktuatif pada kelas minoritas seperti "jute" dan "rice", yang menunjukkan adanya potensi false negative lebih tinggi.

### Rekomendasi
Agar sistem rekomendasi tanaman lebih andal:
1. Random Forest layak digunakan sebagai model utama karena kestabilannya dan konsistensi tinggi di seluruh kelas.
2. Perhatikan performa recall untuk kelas minoritas karena dapat berdampak pada kesalahan rekomendasi (misalnya, tanaman seperti "rice" yang sangat penting bisa tidak direkomendasikan meskipun sesuai kondisi).

### Perbandingan ketiga model:
![alt text](https://github.com/AtikaOktavianti/-Predictive-Analytics/blob/main/perbandingan.png?raw=true)

Gambar 10. Perbandingan Model

### Interpretasi
1. Semua model memiliki performa sangat tinggi, dengan akurasi di atas 98%, menandakan bahwa dataset kemungkinan bersih, seimbang, dan mudah dipisahkan antar kelas.
2. Random Forest menghasilkan akurasi tertinggi (99.55%), sedikit lebih baik dari XGBoost dan SVM.
3. XGBoost sangat mendekati Random Forest, dengan selisih hanya 0.23%. Ini menunjukkan bahwa keduanya bekerja sangat baik untuk dataset ini.
4. SVM sedikit di bawah dua model lainnya, namun tetap dengan akurasi sangat baik.

### Kesimpulan 
1. Random Forest adalah model terbaik dalam eksperimen ini, meskipun keunggulannya hanya sedikit.
2. Ketiga model layak digunakan, tergantung pada kebutuhan:
- Jika mengutamakan akurasi → Random Forest
- Jika butuh model efisien dan kuat terhadap overfitting → XGBoost
- Jika dataset kecil dan linearitas cukup → SVM

## Referensi
1. Esairina, E. (2023). Memahami confusion matrix: Accuracy, precision, recall, specificity, dan F1-score. Medium. https://esairina.medium.com/memahami-confusion-matrix-accuracy-precision-recall-specificity-dan-f1-score-610d4f0db7cf
2. Kurniawan, M. A., & Falentina, A. T. (2024). Comparison of Support Vector Machine (SVM), XGBoost, and Random Forest for Sentiment Analysis. Proxies: Journal of Informatics, 6(1), 45–52. https://journal.unika.ac.id/index.php/proxies/article/view/12453
3. Permata, A. H. (2024). Evaluasi performa model gradient boosting dengan dan tanpa pruning. Jurnal Informatika Teknologi dan Sains (JINTEKS), 9(2), 120–127. https://jurnal.uts.ac.id/index.php/JINTEKS/article/download/4702/2203/15823
4. Sudirwo, S. E., Hadi, A., Judijanto, L., Purwandari, N., Zain, N. N. L. E., Rambe, K. H., Mukhlis, I. R., Jihadi, H., Mahliatussikah, H., Baskoro, B. H., & Yusufi, A. (2025). Artificial intelligence: Teori, konsep, dan implementasi di berbagai bidang. ResearchGate. https://www.researchgate.net/publication/391369694_ARTIFICIAL_INTELLIGENCE
5. Torhino, R., & Andono, P. (2024). Penerapan algoritma Random Forest dalam prediksi curah hujan untuk mendukung analisis cuaca. Building of Informatics, Technology and Science (BITS), 6(3), 1688–1699. https://doi.org/10.47065/bits.v6i3.6404
