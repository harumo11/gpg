## Reference
- [PointNetGPD](https://github.com/lianghongzhuo/PointNetGPD)
- [gpg](https://github.com/atenpas/gpg)
- [gpd](https://github.com/atenpas/gpd)

## 用語

- $h$ : グリッパーの6-DoFポーズ（姿勢と位置）

  - 数学的にはある瞬間の「グリッパーの姿勢＋位置」を表す
  - 実装上は同時変換行列(4x4)
  - 論文内での表記 $h \in R \in SE(3)$ $R$は探索空間，SE(3)は３次元のポーズ群

- $B(h)$ : グリッパー本体領域 (Body of hand)

  - グリッパーの開いた状態で占める物理ボリューム領域

  - 衝突判定で使う．点群と本体がめり込まないかを判定

  - 点群全体とB(h)の積集合の中に1つでも点群が存在したら，めり込んでいると判定

  - 次元は空間領域（3次元体積）

  - 実装上の表現は

    - OOBB（おおざっぱにやるなら）
    - ボクセル集合
    - メッシュポリゴン（厳密にやるなら）

  - 処理例：
    ```python
    # 例：グリッパー本体B(h)が点群にめり込むか？
    is_collision = np.any( check_points_in_box(point_cloud, B_h_volume) )
    ```

- $C(h)$：グリッパーが閉じる領域

  - グリッパーの指が閉じるときに通過する領域（スイープ体積）

  - 「この空間内に点が存在すること」が「そのハンドポーズが意味のある候補」の最低条件

  - 次元は空間領域（3次元体積）

  - 具体例：

    - パラレルグリッパーなら「指が開いた位置から閉じるまでの間の隙間全体」
    - 幅×高さ×奥行きの長方体領域などで近似できる

  - 処理例：
    ```python
    # 例：C(h)内に点があるか？
    points_in_C_h = extract_points_in_box(point_cloud, C_h_volume)
    if len(points_in_C_h) == 0:
        continue  # 候補として不適
    ```

    
