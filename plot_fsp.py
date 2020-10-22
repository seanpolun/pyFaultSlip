import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Parameters
rot_angle = 75.
sv_depth = 130.
#############

data = pd.read_csv('./revised_fsp_export.csv')
data.rename(columns={' CenterX': 'X1', ' CenterY': 'Y1'}, inplace=True)
data.rename(columns={' Strike(deg)': 'Strike(deg)', ' Dip(Deg)': 'Dip(deg)',' length(km)': 'length(km)'}, inplace=True)
data['X2'] = (np.sin(np.deg2rad(data['Strike(deg)'])) * data['length(km)']) + data.X1
data['Y2'] = (np.cos(np.deg2rad(data['Strike(deg)'])) * data['length(km)']) + data.Y1
fig, ax = plt.subplots(dpi=300)
xmin = data.X1.min(axis=0)
xmax = data.X1.max(axis=0)
ymin = data.Y1.min(axis=0)
ymax = data.Y1.max(axis=0)
rot_angle = 75.

increase_scale = 0.25
xrange = xmax - xmin
yrange = ymax - ymin
xinc = (xrange * increase_scale) / 2
yinc = (yrange * increase_scale) / 2
xmin1 = xmin - xinc
xmax1 = xmax + xinc
ymin1 = ymin - yinc
ymax1 = ymax + yinc
prob_ind = '0.33'
plotmin = (data[prob_ind].min(axis=0) / 145.03)
plotmax = (data[prob_ind].max(axis=0) / 145.03)
plotmin1 = plotmin - 5
plotmax1 = plotmax + 5
# plotmin1 = 0
# plotmax1 = 50.
norm1 = mpl.colors.Normalize(vmin=plotmin1, vmax=plotmax1)
ax.set_xlim(xmin1, xmax1)
ax.set_ylim(ymin1, ymax1)
for index, row in data.iterrows():
    x1 = row['X1']
    y1 = row['Y1']
    y3 = row['Y2']
    x3 = row['X2']
    x2 = x1 + (x1 - x3)
    y2 = y1 + (y1 - y3)

    value = (row[prob_ind] / 145.03)
    vertices = np.array([[x1, y1], [x2, y2], [x3, y3]])
    lc = mpl.collections.LineCollection([vertices], cmap=plt.get_cmap('jet_r'), norm=norm1)
    lc.set_array(np.array([value, value]))
    lc.set_linewidth(2)
    ax.add_collection(lc)
axcb = fig.colorbar(lc)
axcb.set_label('Failure Pressure [MPa]')
ax2 = fig.add_axes([0.15, 0.1, 0.2, 0.2])
str_img = plt.imread('./resources/h_stresses.png')
stress_im = ax2.imshow(str_img)
midx = str_img.shape[0] / 2
midy = str_img.shape[1] / 2
transf = mpl.transforms.Affine2D().rotate_deg_around(midx, midy, rot_angle) + ax2.transData
stress_im.set_transform(transf)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set(frame_on=False)
plt.show()