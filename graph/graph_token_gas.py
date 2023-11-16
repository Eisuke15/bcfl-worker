import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# CSVファイルを読み込みます
df = pd.read_csv('iid.csv')

# グラフを作成します
fig, ax1 = plt.subplots()

# token balanceを棒グラフで表示します
df['token balance'].plot(kind='bar', ax=ax1, color='b', alpha=0.5)
ax1.set_ylabel('token balance')
ax1.set_xlabel('worker')

ax1.set_ylim(bottom=0, top=190)

# gas Consumedを折れ線グラフでその上に表示します
ax2 = ax1.twinx()
df['gas Consumed'].plot(kind='line', marker='o', ax=ax2, color='r')
ax2.set_ylabel('gas Consumed')

# レジェンドを追加します
ax1.legend(["token balance"], loc='lower left')
ax2.legend(["gas Consumed"], loc='lower right')

ax2.set_ylim(bottom=0, top=36000000)


# グラフをPDFファイルとして保存します
pdf = PdfPages('graph.pdf')
pdf.savefig(fig, bbox_inches='tight')
pdf.close()
