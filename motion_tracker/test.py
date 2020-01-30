ta, tb = tracks[:2]
ts = np.arange(len(ta))

i = np.logical_and(ts > 2350, ts < 2400)
ts = np.linspace(0, dur/25, len(ta))

plt.plot(ta[i, 1], ta[i, 0], color='k')
plt.scatter(ta[i, 1], ta[i, 0], c=ts[i], cmap='plasma')
# plt.scatter(tb[i, 0], tb[i, 1], c=ts[i], cmap='viridis')
# plt.scatter(ta[i, 0], ta[i, 1], color='g')
# plt.scatter(tb[i, 0], tb[i, 1], color='b')
plt.colorbar()
plt.gca().set_aspect('equal')
plt.show()
