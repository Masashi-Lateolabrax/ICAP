# ノート

## 実験概要

このレポジトリは、**分散最適化とMuJoCo物理シミュレーション**を用いた**群体ロボット制御**の実験コードです。

## 実験目的

- CMA-ES（進化戦略）を用いてロボットのニューラルネットワークを最適化
- 複数のロボットが協調して餌を巣に運搬する**集団行動**の創発
- 分散最適化システムの動作確認

## 実験設定

### ロボット仕様

- **形状**: 円筒形（半径0.175m、高さ0.1m）
- **駆動**: 二輪差動駆動（最大速度0.8m/s）
- **制御**: 6入力2出力の浅いニューラルネットワーク（パラメータ数29）

### センサー構成

センサには全方位センサと方向センサの2種類があります.

全方位センサは次のように計算されます.

$$$
S_\text{omni} = \alpha \sum_{(x,\theta) \in \text{Targets}} \frac{1}{ \beta \cdot \min(x-x_o, 0) + 1 } \begin{bmatrix} \cos(\theta) \\ \sin(\theta) \end{bmatrix}
$$$

$x$は知覚対象までの距離､$\theta$は知覚対象の方角、$S_\text{omni}$は全方位センサの出力ベクトル、$\alpha$と$\beta$
はセンサのパラメータ､$x_o$は知覚対象との距離のオフセットです。$\alpha$はセンサのゲイン、$\beta$はセンサの感度です。

この$S_\text{omni}$は直接つかわれれず､次のように事前処理がされます.

$$$
\begin{align*}
i_0 &= \frac{
\text{arctan2}(S_\text{omni}[1], S_\text{omni}[0])
}{
0.5 \, \pi
}\\
i_1 &= \frac{1}{ \| S_\text{omni} \|_2 + 1 }
\end{align*}
$$$

方向センサは次のように計算されます.

$$$
S_\text{dir} = \frac{ \theta_\text{Target} }{ 0.5 \, \pi }
$$$

ここで$\theta_\text{Target}$は知覚対象の方角であり､単位はラジアンです。

これらのセンサをロボットは持っており､次のようになってる.

1. **ロボットセンサー**: 全方位センサ. 自身以外のロボットを検出
2. **食物センサー**: 全方位センサ. 食物を検出
3. **方向センサー**: 方向センサ. 巣のある方向を検出

### 最適化パラメータ

- **アルゴリズム**: CMA-ES（共分散行列適応進化戦略）
- **集団サイズ**: 20個体
- **最大世代数**: 1000世代
- **初期ステップサイズ**: 0.5

## 評価関数

$$$
L_\text{R} =
-\sum_{p_r \in \text{Robots}}
\,
\sum_{p_f \in \text{Foods}}
\exp \left( \frac{\| p_r - p_f \|_2^2}{\sigma_r} \right)
$$$

$$$
L_\text{F} =
-\sum_{p_f \in \text{Foods}}
\exp \left( \frac{\| p_f - p_n \|_2^2}{\sigma_f} \right)
$$$

$$$
L = \lambda_r L_\text{R}
+\lambda_f L_\text{F}
+\lambda_\text{reg} \cdot \| \text{Parameters} \|_2
$$$

## 実験手順

1. `bin/server.py`でCMA-ESサーバーを起動

```commandline
name@hostname:/path/to/ICAP$ python bin/server.py
```

2. bin/main.py`でクライアントを接続

```commandline
name@hostname:/path/to/ICAP$ python bin/main.py
```

## 期待される結果

- **主目標**: ロボットが餌を効率的に巣に運搬する協調行動の創発
- **最小成果**: 分散最適化システムの正常動作確認
- **追加観察**: 群体行動パターンの分析と最適化進捗の可視化