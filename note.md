# ノート

## 実験概要

このレポジトリは、**分散最適化とMuJoCo物理シミュレーション**を用いた**群体ロボット制御**の実験コードです。CMA-ES（進化戦略）を用いてロボットのニューラルネットワークを最適化し、複数のロボットが協調して餌を巣に運搬する集団行動の創発を目指します。

## 実験目的

1. **ニューラルネットワーク最適化**: CMA-ES（進化戦略）を用いたロボット制御器の進化
2. **集団行動の創発**: 複数のロボットが協調して餌を巣に運搬する行動の自発的発現
3. **分散最適化システムの実証**: サーバー・クライアント型の分散計算システムの動作確認

## アーキテクチャ

### システム構成

```
bin/server.py  ←→  bin/main.py (クライアント)
     ↓                 ↓
OptimizationServer  Simulation + 評価
     ↓                 ↓
  CMA-ES        MuJoCo物理シミュレーション
```

- **サーバー**: CMA-ESアルゴリズムを実行し、個体パラメータを配布
- **クライアント**: MuJoCo物理シミュレーションを実行し、適応度を評価
- **通信**: TCP/IPソケット通信で分散処理を実現

## 実験設定

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

## 実験手順

1. **サーバー起動**: CMA-ESサーバーを起動

```bash
python bin/server.py
```

2. **クライアント接続**: 物理シミュレーションクライアントを接続

```bash
python bin/main.py
```

3. **実行フロー**:
   - サーバーが個体パラメータを生成・配布
   - クライアントがMuJoCo物理シミュレーションを実行（60秒間）
   - 評価結果をサーバーに送信
   - CMA-ESが次世代パラメータを生成
   - 収束条件まで繰り返し実行

## 期待される結果

- **主目標**: ロボットが餌を効率的に巣に運搬する協調行動の創発
- **最小成果**: 分散最適化システムの正常動作確認  
- **追加観察**: 群体行動パターンの分析と最適化進捗の可視化

## 設定詳細

### デフォルト設定 (`framework/config/_settings.py`)

| パラメータ | デフォルト値 | 説明 |
|-----------|-------------|------|
| `Robot.NUM` | 1 | ロボット数 |
| `Robot.RADIUS` | 0.175m | ロボット半径 |
| `Robot.MAX_SPEED` | 0.8m/s | 最大速度 |
| `Food.NUM` | 1 | 食物数 |
| `Food.RADIUS` | 0.5m | 食物半径 |
| `Nest.RADIUS` | 1.0m | 巣半径 |
| `Simulation.TIME_LENGTH` | 60秒 | シミュレーション時間 |
| `Simulation.TIME_STEP` | 0.01秒 | タイムステップ |
| `Optimization.population_size` | 20 | CMA-ES集団サイズ |
| `Optimization.sigma` | 0.5 | 初期ステップサイズ |

### 実行時設定の上書き

`bin/main.py`では以下の設定が動的に変更される:
- レンダリング解像度: 480×320
- ロボット初期位置: (0, 0, π/2)
- 食物初期位置: (0, 2)

### 損失関数のパラメータ

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `GAIN_ROBOT_AND_FOOD` | 0.01 | ロボット-食物間損失の重み |
| `SIGMA_ROBOT_AND_FOOD` | calc_loss_sigma(1, 0.3) | 距離1mで重み0.3となる分散 |
| `OFFSET_ROBOT_AND_FOOD` | 0.675m | ロボット+食物半径の和 |
| `GAIN_NEST_AND_FOOD` | 1.0 | 食物-巣間損失の重み |
| `SIGMA_NEST_AND_FOOD` | calc_loss_sigma(4, 0.01) | 距離4mで重み0.01となる分散 |
| `OFFSET_NEST_AND_FOOD` | 0 | 食物-巣間のオフセット |

## 実装の特徴

### 分散処理アーキテクチャ

- **サーバー・クライアント型**: 最適化と評価の分離
- **非同期処理**: 複数クライアントの並列実行対応
- **堅牢性**: 接続エラーや個体破損に対する耐性

### センサーの工夫

- **正規化処理**: 全方位センサの出力を角度と強度に分解
- **距離減衰**: 遠い対象の影響を指数関数的に減衰
- **座標変換**: ロボット座標系での一貫した処理

### 最適化の特徴

- **適応的探索**: CMA-ESによる共分散行列の自動調整
- **多目的最適化**: 複数の損失項の重み付け合成
- **正則化**: パラメータの過学習防止

## トラブルシューティング

### よくある問題

1. **接続エラー**: サーバーが起動していない
   - 解決: `python bin/server.py` でサーバーを先に起動

2. **シミュレーション失敗**: MuJoCo環境の問題
   - 解決: 依存関係とMuJoCo設定を確認

3. **最適化の収束不良**: パラメータ設定の問題
   - 解決: `sigma`や損失重みの調整

### ログ出力

- サーバー: 最適化進捗と接続状況
- クライアント: シミュレーション結果と評価値