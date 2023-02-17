import streamlit as st
import pandas as pd
from pomegranate import DiscreteDistribution, HiddenMarkovModel
import matplotlib.pyplot as plt
import hmmlearn.hmm as hmm
import numpy as np

kolom = None
# Membuat form input untuk file csv tahun sebelumnya
file_sebelumnya = st.file_uploader("Upload file csv tahun sebelumnya", type="csv")
# Membuat tombol untuk menampilkan data awal 10 dari file csv tahun sebelumnya
if file_sebelumnya:
    df_sebelumnya = pd.read_csv(file_sebelumnya)
    if st.button("Tampilkan data awal 10 dari file csv tahun sebelumnya"):
        st.write("Data awal 10 dari file csv tahun sebelumnya:")
        st.write(df_sebelumnya.head(10))
# Membuat form input untuk file csv tahun sekarang
file_sekarang = st.file_uploader("Upload file csv tahun sekarang", type="csv")
# Membuat tombol untuk menampilkan data awal 10 dari file csv tahun sekarang
if file_sekarang:
    df_sekarang = pd.read_csv(file_sekarang)
    if st.button("Tampilkan data awal 10 dari file csv tahun sekarang"):
        st.write("Data awal 10 dari file csv tahun sekarang:")
        st.write(df_sekarang.head(10))
# Membuat form input untuk memilih kolom yang akan dicari kata kunci
if file_sebelumnya and file_sekarang:
    kolom = st.selectbox("Pilih kolom yang akan dicari kata kunci", df_sebelumnya.columns)

# Membuat form input untuk memasukkan kata kunci
if kolom!= None:
    keyword = st.text_input("Masukkan kata kunci")

# Membuat tombol untuk menampilkan jumlah keyword yang cocok
if file_sebelumnya and file_sekarang and kolom and keyword:
    if st.button("Tampilkan jumlah keyword"):
        # Menghitung jumlah kemunculan kata kunci pada file csv tahun sebelumnya
        jumlah_sebelumnya = df_sebelumnya[kolom].str.contains(keyword, case=False).sum()

        # Menghitung jumlah kemunculan kata kunci pada file csv tahun sekarang
        jumlah_sekarang = df_sekarang[kolom].str.contains(keyword, case=False).sum()

        # Menampilkan hasil
        st.write(f"Jumlah keyword '{keyword}' pada file csv tahun sebelumnya: {jumlah_sebelumnya}")
        st.write(f"Jumlah keyword '{keyword}' pada file csv tahun sekarang: {jumlah_sekarang}")

if file_sebelumnya and file_sekarang and kolom and keyword:
    if st.button("Prediksi data untuk tahun berikutnya"):
        # Membaca data tahun sebelumnya dan tahun sekarang
        data = pd.concat([df_sebelumnya[kolom], df_sekarang[kolom]], ignore_index=True)
        #data_bersih = pd.concat([df_sebelumnya, df_sekarang],ignore_index=True)
        #st.write(data_bersih.head(10))
        # Menghitung frekuensi kemunculan setiap kata pada data
        freq = {}
        for d in data:
            d_str = str(d)
            for w in d_str.split():
                freq[w] = freq.get(w, 0) + 1

        # Membuat model HMM
        states = list(freq.keys())
        
        # Menghitung total jumlah baris pada tabel yang cocok dengan keyword
        total_data_sebelumnya = df_sebelumnya[kolom].str.contains(keyword, case=False).sum()
        total_data_sekarang = df_sekarang[kolom].str.contains(keyword, case=False).sum()

        # Membuat tabel dengan no, keyword, total judul yang cocok, dan total data yang cocok
        no = []
        keyword_list = []
        tb_data_sebelumnya = []
        tb_data_sekarang = []
        total_data = []

        for i, s in enumerate(states):
            if keyword in s:
                no.append(i+1)
                keyword_list.append(s)
                tb_data_sebelumnya.append(sum(df_sebelumnya[kolom].str.contains(s, case=False).dropna()))
                tb_data_sekarang.append( sum(df_sekarang[kolom].str.contains(s, case=False).dropna()))
                #total_data.append(total_data_sebelumnya + total_data_sekarang)
                total_data.append(sum(df_sebelumnya[kolom].str.contains(s, case=False).dropna()) + sum(df_sekarang[kolom].str.contains(s, case=False).dropna()))

        df = pd.DataFrame({"no": range(1, len(keyword_list)+1), "keyword": keyword_list, "data tahun sebelumnya": tb_data_sebelumnya, "data tahun sekarang": tb_data_sekarang, "total data": total_data})
        # tambahkan kolom total data
        df.loc[len(df)] = ['', 'Total', sum(df['data tahun sebelumnya']), sum(df['data tahun sekarang']),sum(df['total data'])]
        st.write(df)    
        X = df[['data tahun sebelumnya', 'data tahun sekarang', 'total data']].to_numpy()

        # Inisialisasi Model HMM
        n_components = min(1, len(df))
        st.write(n_components)
        model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=160)

        # Mengatur nilai awal transisi antar-states
        model.transmat_ = np.ones((n_components, n_components)) / n_components

        # Melatih Model HMM
        model.fit(X)

        # Normalisasi matriks transisi
        model.transmat_ = model.transmat_ / model.transmat_.sum(axis=1)[:, np.newaxis]

        # Prediksi Peminjaman Buku Tahun Berikutnya
        next_year_data = np.array([[200, 300, 500]])

        predicted_data = []
        for i in range(next_year_data.shape[1]):
            next_month_data = next_year_data[:, i].reshape(-1, 1)
            predicted_state = model.predict(next_month_data)
            predicted_data.append(model.means_[predicted_state][0][0])

        #st.write(predicted_data)

        # Membuat grafik prediksi menggunakan HMM
        plt.figure(figsize=(12, 6))
        plt.title("Grafik prediksi menggunakan HMM")
        plt.xlabel("Tahun")
        plt.ylabel("Jumlah data yang cocok")
        plt.xticks([0, 1, 2], ["Tahun Sebelumnya", "Tahun Sekarang", "Tahun Berikutnya"])
        plt.bar([0, 1, 2], [total_data_sebelumnya, total_data_sekarang, predicted_data[2]])
        plt.show()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        # Menampilkan grafik
        st.pyplot()
        st.write(f"Pinjaman buku untuk tahun berikutnya dari keyword: '{keyword}' adalah '{predicted_data[2]}'")
        