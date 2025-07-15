# ノート

## 実験概要

このレポジトリは、**分散最適化とMuJoCo物理シミュレーション**を用いた**群体ロボット制御**
の実験コードです。CMA-ES（進化戦略）を用いてロボットのニューラルネットワークを最適化し、複数のロボットが協調して餌を巣に運搬する集団行動の創発を目指します。

## 実験目的

1. STLファイルをサポートしたため, その動作確認

## 実験設定

### タスク設定

餌を巣に運搬すること.

### ロボット仕様

- **形状**: 円筒形（半径0.175m、高さ0.1m）
- **駆動**: 二輪差動駆動（最大速度0.8m/s）
- **制御**: 6入力2出力の浅いニューラルネットワーク
    - 構造: 6→3→2（隠れ層：Tanhshrink、出力層：Tanh）
    - パラメータ数: 29個（重み+バイアス）

### センサー構成

ロボットは3つのセンサを持ちます：

1. **ロボットセンサー**: 前処理済み全方位センサ。自身以外のロボットを検出
2. **食物センサー**: 前処理済み全方位センサ。食物を検出
3. **方向センサー**: 方向センサ。巣のある方向を検出

#### 全方位センサ (OmniSensor)

知覚対象の位置から出力ベクトルを計算：

$$$
\mathbf{S}_{\text{omni}} = \alpha \sum_{i \in \text{Targets}} \frac{1}{\beta \cdot \max(d_i - d_{\text{offset}}, 0) + 1} \begin{bmatrix} \cos(\theta_i) \\ \sin(\theta_i) \end{bmatrix}
$$$

ここで：

- $d_i$: 知覚対象 $i$ までの距離
- $\theta_i$: 知覚対象 $i$ の方向角（ロボット座標系）
- $\alpha$: 出力ゲイン
- $\beta$: 距離ゲイン
- $d_{\text{offset}}$: 距離オフセット

#### 前処理済み全方位センサ (PreprocessedOmniSensor)

全方位センサの出力を正規化：

$$$
\begin{align}
\text{magnitude} &= \frac{1}{\|\mathbf{S}_{\text{omni}}\|_2 + 1} \\
\text{angle} &= \frac{\arctan2(\mathbf{S}_{\text{omni}}[1], \mathbf{S}_{\text{omni}}[0])}{\pi/2}
\end{align}
$$$

出力: $[\text{magnitude}, \text{angle}]$

#### 方向センサ (DirectionSensor)

ターゲットへの相対方向を計算：

$$$
\mathbf{S}_{\text{dir}} = \frac{\arctan2(\mathbf{u}_{\text{target}} \cdot \mathbf{e}_y, \mathbf{u}_{\text{target}} \cdot \mathbf{e}_x)}{\pi/2}
$$$

ここで：

- $\mathbf{u}_{\text{target}}$: ターゲットへの単位ベクトル
- $\mathbf{e}_x, \mathbf{e}_y$: ロボット座標系の基底ベクトル

### 最適化パラメータ

- **アルゴリズム**: CMA-ES（共分散行列適応進化戦略）
- **集団サイズ**: 20個体
- **最大世代数**: 1000世代
- **初期ステップサイズ**: 0.5
- **パラメータ次元**: 29（ニューラルネットワークの重み・バイアス）

## 評価関数

損失関数は3つの成分で構成されます：

### ロボット-食物間の距離損失

$$$
L_{\text{robot-food}} = \lambda_{\text{rf}} \sum_{r \in \text{Robots}} \sum_{f \in \text{Foods}} \exp\left(-\frac{\max(\|\mathbf{p}_r - \mathbf{p}_f\|_2 - d_{\text{offset}}, 0)^2}{\sigma_{\text{rf}}}\right)
$$$

### 食物-巣間の距離損失

$$$
L_{\text{food-nest}} = \lambda_{\text{fn}} \sum_{f \in \text{Foods}} \exp\left(-\frac{\max(\|\mathbf{p}_f - \mathbf{p}_n\|_2 - d_{\text{offset}}, 0)^2}{\sigma_{\text{fn}}}\right)
$$$

### 正則化項

$$$
L_{\text{reg}} = \lambda_{\text{reg}} \|\mathbf{\theta}\|_2
$$$

### 総損失

$$$
L_{\text{total}} = L_{\text{robot-food}} + L_{\text{food-nest}} + L_{\text{reg}}
$$$

ここで：

- $\mathbf{p}_r$: ロボット $r$ の位置
- $\mathbf{p}_f$: 食物 $f$ の位置
- $\mathbf{p}_n$: 巣の位置
- $\mathbf{\theta}$: ニューラルネットワークのパラメータ
- $\lambda_{\text{rf}}, \lambda_{\text{fn}}, \lambda_{\text{reg}}$: 各損失の重み
- $\sigma_{\text{rf}}, \sigma_{\text{fn}}$: 各損失の分散パラメータ

## 期待される結果

- **主目標**: 餌が巣に運搬される

# Method

## Simulation Environment

The simulation uses MuJoCo with the following robot specifications:

- **Sensors**: Two omni-sensors (for robots and food detection)
- **Actuators**: Two wheels for movement
- **Communication**: Pheromone secretion organ
- **Control**: Artificial neural network controller

### Omni-Sensor Formula

The omni-sensor aggregates positional information using:

$$
\mathrm{OmniSensor} = \frac{1}{N_\mathrm{O}} \sum_{o \in \mathrm{Objects}}
\frac{1}{d_o + 1}
\begin{bmatrix}
\cos \theta_o \\
\sin \theta_o
\end{bmatrix}
$$

Where:

- *Objects*: Set of sensing targets
- $N_\mathrm{O}$: Number of objects
- $d_o$: Distance between robot and object $o$
- $\theta_o$: Relative angle from robot to object $o$

## Task Description

Robots operate in a rectangular field environment where:

- Food items are randomly distributed
- A nest is positioned at the field center
- Food respawns randomly when collected
- **Goal**: Transport food to the nest efficiently

## Optimization

The system uses CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for neural network parameter optimization.