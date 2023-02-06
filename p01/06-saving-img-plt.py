import matplotlib.pyplot as plt

plt.plot([1, 2, 3], [100, 200, 300])
plt.title('Sebuah "Judul"')
plt.xlabel("Tahun")
plt.ylabel("Jumlah Mahasiswa")
plt.savefig("../images/grafik_mhs.png")
plt.show()
