# Frozen N=20 Protocol

N=20 is the final planned pre-GAT TRAIN-development expansion. Residual calibration invalidity leads to a methodology decision, with no automatic proposal or acquisition beyond twenty scenarios.

Frozen before network access: acquisition config SHA-256 `e55a6a18261832c5b6329e0e84f07866cd2f856a9d0f8e0a5b69341d4d445600`; compute-reuse policy SHA-256 `9d6f188e478ef7376e0730df26018da4519166fd0ef467e7530bc7a8481031d2`. First twenty complete official TRAIN scenarios in parsed archive order, whole scenarios, PCG64 seed 2026, exactly 15 FIT / 5 CAL, window 12, unchanged feature/normality/calibration/audit/recipe code and default severities. No performance-dependent selection.

# Network / Prefix Integrity

N=10 prefix before/after: 12,547,522,560 bytes, SHA-256 `c565f5f416b264b7725c15e79d0fe7bee92e8439866647826d8c8e58afb02be0`. A real independent N=20 staging copy was verified before appending. Original 3 GB prefix is also byte-hash verified unchanged. No N=10/N=2 data or artifacts were overwritten.

One sequential N=10-prefix replay reconstructed the live gzip/TAR state because the previous process's zlib state could not safely be serialized. HTTP 206 Range resumed at byte 12,547,522,560, with exact offset/length/encoding checks and stable validator '"6a200b7a-22194ff7f3"'. No initial-prefix redownload or independent later-range decompression. Added network bytes: 13,657,494,800; final N=20 prefix: 26,205,017,360 bytes, SHA-256 `015ef22f4312b70336f92214ac069979df9c57a42c26e0aec913863917961698`.

Stopped immediately after twenty complete scenarios at `{"kind": "subsequent_scenario_header", "next_scenario_id": "Town02/scenario-5", "tar_header_offset": 28207053312, "member": "train/Town02/scenario-5/rgb-front"}`. At most one 256 KiB compressed chunk contains boundary read-ahead. The trailing scenario was not counted or extracted. Acquisition log: `C:\Users\Aditya\AppData\Local\Temp\carla_train_expand20/acquisition_log.jsonl`.

# 20 Complete TRAIN Scenarios

| Ordinal | Scenario | Town | RGB / Seg count and range | GNSS / IMU rows | Reused raw | Partition |
|---|---|---|---|---|---|---|
| 1 | Town01/scenario-1 | Town01 | 3000/3000, [0, 2999] | 3000/3000 | True | CAL_NORMAL |
| 2 | Town01/scenario-10 | Town01 | 3000/3000, [0, 2999] | 3000/3000 | True | FIT_NORMAL |
| 3 | Town01/scenario-11 | Town01 | 3000/3000, [0, 2999] | 3000/3000 | True | FIT_NORMAL |
| 4 | Town01/scenario-12 | Town01 | 3000/3000, [0, 2999] | 3000/3000 | True | FIT_NORMAL |
| 5 | Town01/scenario-13 | Town01 | 3000/3000, [0, 2999] | 3000/3000 | True | FIT_NORMAL |
| 6 | Town01/scenario-14 | Town01 | 3000/3000, [0, 2999] | 3000/3000 | True | FIT_NORMAL |
| 7 | Town01/scenario-15 | Town01 | 3000/3000, [0, 2999] | 3000/3000 | True | FIT_NORMAL |
| 8 | Town01/scenario-2 | Town01 | 3000/3000, [0, 2999] | 3000/3000 | True | FIT_NORMAL |
| 9 | Town01/scenario-3 | Town01 | 3000/3000, [0, 2999] | 3000/3000 | True | FIT_NORMAL |
| 10 | Town01/scenario-4 | Town01 | 3000/3000, [0, 2999] | 3000/3000 | True | CAL_NORMAL |
| 11 | Town01/scenario-5 | Town01 | 3000/3000, [0, 2999] | 3000/3000 | False | FIT_NORMAL |
| 12 | Town01/scenario-6 | Town01 | 3000/3000, [0, 2999] | 3000/3000 | False | FIT_NORMAL |
| 13 | Town01/scenario-7 | Town01 | 3000/3000, [0, 2999] | 3000/3000 | False | FIT_NORMAL |
| 14 | Town01/scenario-8 | Town01 | 3000/3000, [0, 2999] | 3000/3000 | False | CAL_NORMAL |
| 15 | Town01/scenario-9 | Town01 | 3000/3000, [0, 2999] | 3000/3000 | False | CAL_NORMAL |
| 16 | Town02/scenario-1 | Town02 | 3000/3000, [0, 2999] | 3000/3000 | False | FIT_NORMAL |
| 17 | Town02/scenario-10 | Town02 | 3000/3000, [0, 2999] | 3000/3000 | False | FIT_NORMAL |
| 18 | Town02/scenario-2 | Town02 | 3000/3000, [0, 2999] | 3000/3000 | False | FIT_NORMAL |
| 19 | Town02/scenario-3 | Town02 | 3000/3000, [0, 2999] | 3000/3000 | False | CAL_NORMAL |
| 20 | Town02/scenario-4 | Town02 | 3000/3000, [0, 2999] | 3000/3000 | False | FIT_NORMAL |

Every scenario has complete consumed members, contiguous aligned RGB/Seg coverage, readable required Base feather tables, aligned sensor indexes and finite required GNSS/IMU values, strict TRAIN loader validation and proven subsequent-scenario completion evidence. The frozen dataset manifest records first TAR offsets, all feather names/schemas/counts, actual paths and per-source-file sizes/SHA-256 hashes.

# Town Diversity

Unique towns: 2. Scenarios per town: `{"Town01": 15, "Town02": 5}`. Selection was retained exactly in archive order.

This checkpoint includes the recorded towns only; it makes no held-out town or final generalization claim.

# Frozen 15/5 FIT-CAL Manifest

Dataset manifest SHA-256 `5f401224e65ab581af7d7558c65b64e149af863bfbe21023968906d3f1066ddd`. Frozen after all twenty scenarios validated, before agent fitting. Sorted canonical IDs and existing PCG64 seed 2026 rule were applied once to the full list; N=10 membership was not manually extended.

FIT_NORMAL: Town01/scenario-10, Town01/scenario-11, Town01/scenario-12, Town01/scenario-13, Town01/scenario-14, Town01/scenario-15, Town01/scenario-2, Town01/scenario-3, Town01/scenario-5, Town01/scenario-6, Town01/scenario-7, Town02/scenario-1, Town02/scenario-10, Town02/scenario-2, Town02/scenario-4.

CAL_NORMAL: Town01/scenario-1, Town01/scenario-4, Town01/scenario-8, Town01/scenario-9, Town02/scenario-3.

| Earlier scenario | N=10 role | N=20 role |
|---|---|---|
| Town01/scenario-1 | FIT_NORMAL | CAL_NORMAL |
| Town01/scenario-10 | CAL_NORMAL | FIT_NORMAL |
| Town01/scenario-11 | FIT_NORMAL | FIT_NORMAL |
| Town01/scenario-12 | FIT_NORMAL | FIT_NORMAL |
| Town01/scenario-13 | FIT_NORMAL | FIT_NORMAL |
| Town01/scenario-14 | FIT_NORMAL | FIT_NORMAL |
| Town01/scenario-15 | FIT_NORMAL | FIT_NORMAL |
| Town01/scenario-2 | FIT_NORMAL | FIT_NORMAL |
| Town01/scenario-3 | FIT_NORMAL | FIT_NORMAL |
| Town01/scenario-4 | CAL_NORMAL | CAL_NORMAL |

# N=20 Compact Cache

One production build/assembly: 59,980 clean rows (44,985 FIT / 14,995 CAL), plus 14,995 CAL-derived pseudo rows. Schema 2, content SHA-256 `c2df480a712e819d85437a9bb141bca91b47b68e90ae84fcf78d9151b93b5539`. NPZ: 52,787,238 bytes. Measured preprocessing/assembly: 80.63 minutes.

Reused 29,990 clean rows across all first ten scenarios exactly, after code/window/schema/source hash and partition-independence checks. Only the ten newly acquired scenarios underwent clean raw feature extraction. Production builder ordering, serialization, manifest construction and content hashing are unchanged; orchestration returns verified old feature blocks with N=20 partition labels.

Reused CAL pseudo scenarios: ['Town01/scenario-4'] (2,999 rows). Excluded prior CAL→FIT pseudo scenarios: ['Town01/scenario-10']. Other N=20 CAL pseudo rows were generated by unchanged production recipes. Final old clean features are byte-identical; eligible pseudo rows equal their old arrays exactly. No CAL pseudo enters FIT graph training, and no graph rows were generated.

Verified all scenario/tick/partition counts, finite feature blocks, schema/content hashes, CAL-only pseudo parentage, target_normal semantics, causal parent windows and exact untouched modality blocks. No second deterministic rebuild.

# Four-Agent Calibration Validity

| Agent | FIT scenarios / rows | CAL scenarios / clean / pseudo | Optimizer converged | Gradient norm | Min Hessian eigenvalue | Pooled structure | Boundary trend | finite_optimum / valid |
|---|---|---|---|---|---|---|---|---|
| Camera | 15/44985 | 5/14995/4285 | True | 2.35864809e-17 | 0.0718792868 | overlapping | not_decreasing | True/True |
| Seg | 15/44985 | 5/14995/2145 | True | 1.36990694e-16 | 0.0145315273 | overlapping | not_decreasing | True/True |
| GNSS | 15/44985 | 5/14995/4285 | True | 0.106969261 | -0.000284101783 | quasi_separated | decreasing | False/False |
| IMU | 15/44985 | 5/14995/4280 | True | 1.04125982e-16 | 0.00139987609 | overlapping | not_decreasing | True/True |

| Agent | Mean normal / pseudo range | Pooled normal / pseudo range | Pooled overlap | Best mean threshold | Violations / observations | Numerical status |
|---|---|---|---|---|---|---|
| Camera | [0.0107769105, 0.849546598] / [8.46240754e-12, 0.840889907] | [0.00955291717, 0.850321496] / [5.47414941e-12, 0.841963095] | [0.00955291717, 0.841963095] | 0.0266411563 | 2577/19280 | finite_stationary_candidate |
| Seg | [4.10000042e-06, 0.950531872] / [0, 0.899092943] | [7.03633815e-07, 0.950919307] / [0, 0.899711556] | [7.03633815e-07, 0.899711556] | 0.0102403975 | 910/17140 | finite_stationary_candidate |
| GNSS | [3.67464712e-16, 0.954263447] / [0, 0] | [0, 0.954482806] / [0, 0] | [0, 0] | 1.83732356e-16 | 0/19280 | suspected_separation |
| IMU | [2.78920025e-23, 0.980114765] / [0, 0.787116691] | [2.73376153e-28, 0.980474751] / [0, 0.788878764] | [2.73376153e-28, 0.788878764] | 1.02003405e-05 | 760/19275 | finite_stationary_candidate |

Thresholds are diagnostic only. Calibration objective, positive-slope parameterization, optimizer and validity audit are unchanged. Convergence alone does not establish validity; the report retains gradients, curvature, structural geometry, boundary/profile diagnostics and the complete audit in calibration_results.json. Accepted finite stationary candidates are numerical evidence, without a global-existence claim. Invalid scientific probability/UQ remains null.

# Camera N=2→10→20

| Agent | N | Mean geometry | Member geometry | Mean overlap | Violations | Valid | Status |
|---|---|---|---|---|---|---|---|
| Camera | 2 | overlapping | overlapping | [1.63723118e-11, 3.59674257e-06] | 22/3856 (0.5705%) | False | suspected_separation |
| Camera | 10 | overlapping | overlapping | [9.31296501e-08, 0.741593026] | 859/7712 (11.1385%) | False | unvalidated_optimizer_result |
| Camera | 20 | overlapping | overlapping | [0.0107769105, 0.840889907] | 2577/19280 (13.3662%) | True | finite_stationary_candidate |

Camera reaches an accepted finite stationary calibration at N=20 under the unchanged protocol: 6 optimizer iterations, raw slope 2.96127179, gradient 2.35864809e-17, minimum curvature 0.0718792868, boundary trend not_decreasing. This is development evidence on the frozen constructed clean-vs-pseudo task, with no final generalization claim.

Finite calibration validity does not establish strong corruption discrimination: normal/pseudo probability medians are 0.815047234/0.819725441, with substantial overlapping score distributions and the diagnostic threshold violations reported above.

# GNSS N=2→10→20

| Agent | N | Mean geometry | Member geometry | Mean overlap | Violations | Valid | Status |
|---|---|---|---|---|---|---|---|
| GNSS | 2 | completely_separated | completely_separated | none | 0/3856 (0.0000%) | False | suspected_separation |
| GNSS | 10 | completely_separated | completely_separated | none | 0/7712 (0.0000%) | False | suspected_separation |
| GNSS | 20 | completely_separated | quasi_separated | none | 0/19280 (0.0000%) | False | suspected_separation |

| CAL class | Member entries / exact zeros | All-zero observations | Squared distance range | Distance / scale range | Exp argument range | Smallest positive score |
|---|---|---|---|---|---|---|
| normal | 74975/1 (0.00133378% zero) | 0 | [0.652297708, 21333.3977] | [0.0931712993, 3057.00898] | [-1528.50449, -0.0465856496] | 6.36931838e-129 |
| pseudo | 21425/21425 (100% zero) | 4285 | [140961949, 654621063] | [20096136.4, 93459502.3] | [-46729751.2, -10048068.2] | none |

Unchanged score: exp(-0.5*d2/d2_scale). Diagnostic distances use the fitted members' existing mahalanobis_distance2 method; exact scalar score reproduction was verified. Float64 smallest positive subnormal=5e-324, smallest positive normal=2.2250738585072014e-308, log(smallest subnormal)=-744.440072. Per-member scales: [6.978519796427494, 7.0043285765546015, 6.9991580698250395, 7.0143806084123925, 7.00105840109806]. Full p10/median/p90 distance and exponent statistics are saved in gnss_distance_diagnostics.json.

normal: 1 exponent arguments fall below log(smallest positive subnormal); all recorded exact-zero entries explained by this underflow condition=True.

pseudo: 21425 exponent arguments fall below log(smallest positive subnormal); all recorded exact-zero entries explained by this underflow condition=True.

Increasing TRAIN/CAL scenario diversity from N=2 → N=10 → N=20 did not remove the GNSS separation pathology. At N=20 every pseudo member score remains exactly zero, and every normal ensemble mean remains strictly positive. Pooled-member geometry changes from complete separation at N=2/N=10 to quasi_separated at N=20: 1 of 74975 normal member entries also underflow to zero. This is an endpoint tie, not substantial class overlap; no normal observation has all five member scores zero.

For these stored scores there is no finite unregularized ensemble-predictive calibration MLE: at any finite intercept, increasing the positive slope leaves every all-zero pseudo probability unchanged while strictly increasing every normal ensemble probability, since each normal observation has at least one positive member score. The mathematical likelihood therefore continues to improve with slope. Numerical convergence does not override the unchanged invalidity audit. Large Mahalanobis distances drive the quantified float64 underflow; no explicit score clamp was used. Recovering numerical score resolution alone would not guarantee removal of structural separation.

# Seg / IMU Stability

| Agent | N | Mean geometry | Member geometry | Mean overlap | Violations | Valid | Status |
|---|---|---|---|---|---|---|---|
| Seg | 2 | overlapping | overlapping | [4.78632631e-25, 0.72283724] | 159/3428 (4.6383%) | True | finite_stationary_candidate |
| Seg | 10 | overlapping | overlapping | [1.15444646e-06, 0.851765938] | 298/6856 (4.3466%) | True | finite_stationary_candidate |
| Seg | 20 | overlapping | overlapping | [4.10000042e-06, 0.899092943] | 910/17140 (5.3092%) | True | finite_stationary_candidate |
| IMU | 2 | overlapping | overlapping | [7.65347924e-05, 0.671071552] | 139/3855 (3.6057%) | True | finite_stationary_candidate |
| IMU | 10 | overlapping | overlapping | [3.61007632e-07, 0.770948973] | 294/7710 (3.8132%) | True | finite_stationary_candidate |
| IMU | 20 | overlapping | overlapping | [2.78920025e-23, 0.787116691] | 760/19275 (3.9429%) | True | finite_stationary_candidate |

| Agent | N | CAL class | prob_normal p10 / median / p90 / max |
|---|---|---|---|
| Seg | 2 | normal | unavailable/0.997279282/unavailable/unavailable |
| Seg | 2 | pseudo | unavailable/0.293935827/unavailable/unavailable |
| Seg | 10 | normal | 0.945466853/0.992896108/0.998491618/0.999323854 |
| Seg | 10 | pseudo | 0.147065583/0.147067207/0.83778847/0.998494177 |
| Seg | 20 | normal | 0.880010096/0.989031788/0.995598487/0.997358659 |
| Seg | 20 | pseudo | 0.294059916/0.294080002/0.895614936/0.996186204 |
| IMU | 2 | normal | unavailable/1/unavailable/unavailable |
| IMU | 2 | pseudo | unavailable/0.0972391781/unavailable/unavailable |
| IMU | 10 | normal | 0.912529493/0.999998182/0.999998195/0.999998533 |
| IMU | 10 | pseudo | 0.106575722/0.106575722/0.550219695/0.999958519 |
| IMU | 20 | normal | 0.916272129/0.999994413/0.999994425/0.999995498 |
| IMU | 20 | pseudo | 0.112945957/0.112945957/0.57147385/0.9999237 |

N=2 accepted results retain historical medians only; unsaved probability quantiles are marked unavailable. N=10/N=20 have full saved distributions. These changes also reflect the recomputed scenario-level split and different CAL composition; the comparison is descriptive, without hyperparameter changes or an isolated causal claim.

# UQ N=20

Exactly preserved: total=H(mean p_t), aleatoric=mean H(p_t), epistemic=max(0,total−aleatoric). Entropy uses the production natural-log convention. Only accepted valid mappings contribute probability-space UQ; invalid mappings remain unavailable.

| Valid agent | CAL class | prob_normal p10 / median / p90 / max | Epistemic p10 / median / p90 / max | Total p10 / median / p90 / max | Aleatoric p10 / median / p90 / max |
|---|---|---|---|---|---|
| Camera | normal | 0.694336131/0.815047234/0.895070062/0.913409838 | 2.45169844e-07/4.33855139e-06/1.00015498e-05/3.46604058e-05 | 0.335782055/0.478821092/0.615587178/0.69314513 | 0.335781429/0.478814603/0.615583671/0.693135629 |
| Camera | pseudo | 0.460150124/0.819725441/0.890604525/0.911360601 | 1.99840144e-15/4.10442867e-07/2.60217854e-06/1.24971549e-05 | 0.345249561/0.471809559/0.689972841/0.693147178 | 0.34524696/0.471808285/0.689972841/0.693142727 |
| Seg | normal | 0.880010096/0.989031788/0.995598487/0.997358659 | 5.35261697e-08/1.05200727e-06/3.96260326e-05/0.0184221947 | 0.0282735737/0.0604046809/0.366904874/0.693147172 | 0.028273438/0.0604039201/0.366803049/0.693029475 |
| Seg | pseudo | 0.294059916/0.294080002/0.895614936/0.996186204 | 0/7.11881665e-11/3.08673456e-05/0.00151898826 | 0.334612175/0.605746949/0.676105182/0.693146982 | 0.334596664/0.605746949/0.676051859/0.693131802 |
| IMU | normal | 0.916272129/0.999994413/0.999994425/0.999995498 | 3.77206875e-11/3.82304459e-11/2.28837561e-05/0.00202881286 | 7.30163671e-05/7.31580791e-05/0.287779138/0.693146808 | 7.30163292e-05/7.31580409e-05/0.287611636/0.693094575 |
| IMU | pseudo | 0.112945957/0.112945957/0.57147385/0.9999237 | 0/0/0.000120511569/0.00033204245 | 0.352630588/0.352630588/0.650751571/0.693097073 | 0.352630588/0.352630588/0.650623434/0.69298188 |

| Agent | N | CAL class | Epistemic p10 / median / p90 / max | Total median | Aleatoric median | Raw score std p10 / median / p90 / max | Probability std p10 / median / p90 / max |
|---|---|---|---|---|---|---|---|
| Camera | 20 | normal | 2.45169844e-07/4.33855139e-06/1.00015498e-05/3.46604058e-05 | 0.478821092 | 0.478814603 | 0.000728453987/0.00247886635/0.00352275491/0.00594016028 | 0.0002296447/0.00114141915/0.00192206593/0.00394110271 |
| Camera | 20 | pseudo | 1.99840144e-15/4.10442867e-07/2.60217854e-06/1.24971549e-05 | 0.471809559 | 0.471808285 | 4.32092125e-08/0.000875765742/0.00208558647/0.00398599254 | 3.17853628e-08/0.000308598118/0.000830562396/0.00211756589 |
| Seg | 2 | normal | unavailable/1.16834373e-05/unavailable/unavailable | 0.0187879133 | 0.0187394404 | 0.00446438422/0.0113040028/0.0227473126/0.082778876 | unavailable |
| Seg | 2 | pseudo | unavailable/6.08291195e-13/unavailable/unavailable | 0.605637942 | 0.605637942 | 0/2.35590684e-07/0.00832735274/0.0361042308 | unavailable |
| Seg | 10 | normal | 9.29521057e-08/3.12476216e-06/7.44149308e-05/0.00318025146 | 0.0422223528 | 0.0422185277 | 0.00099200266/0.00331678023/0.00721367632/0.0245290992 | 1.85073089e-05/0.000201779179/0.00252226734/0.0389503828 |
| Seg | 10 | pseudo | 0/1.02992614e-12/6.65905303e-05/0.00112734246 | 0.417585094 | 0.417585094 | 0/4.18127966e-07/0.00323904702/0.0117998309 | 0/5.08302533e-07/0.00502791663/0.0198039614 |
| Seg | 20 | normal | 5.35261697e-08/1.05200727e-06/3.96260326e-05/0.0184221947 | 0.0604046809 | 0.0604039201 | 0.000693463115/0.00205650097/0.00503844832/0.0602663859 | 2.25067641e-05/0.00014721423/0.00306374517/0.0914723231 |
| Seg | 20 | pseudo | 0/7.11881665e-11/3.08673456e-05/0.00151898826 | 0.605746949 | 0.605746949 | 1.68914765e-88/3.65540525e-06/0.00267863382/0.0402701007 | 0/5.43687612e-06/0.00353440945/0.0179549996 |
| IMU | 2 | normal | unavailable/0/unavailable/unavailable | 2.86304099e-11 | 2.86304099e-11 | 0.000521882218/0.000526155212/0.0166557811/0.0272990415 | unavailable |
| IMU | 2 | pseudo | unavailable/0/unavailable/unavailable | 0.318974131 | 0.318974131 | 0/0/0.00622866655/0.0193473179 | unavailable |
| IMU | 10 | normal | 3.48594428e-11/3.71954513e-11/9.532678e-05/0.00492399859 | 2.58447139e-05 | 2.58446767e-05 | 0.000378035918/0.000403538078/0.00731059955/0.0178650319 | 1.12319345e-08/1.16236844e-08/0.00276407453/0.045007934 |
| IMU | 10 | pseudo | 0/0/0.000282082261/0.00276609709 | 0.339295619 | 0.339295619 | 0/0/0.00302012505/0.0146604096 | 1.38777878e-17/1.38777878e-17/0.0118021787/0.03088797 |
| IMU | 20 | normal | 3.77206875e-11/3.82304459e-11/2.28837561e-05/0.00202881286 | 7.31580791e-05 | 7.31580409e-05 | 0.000250010591/0.00025246133/0.00323456124/0.0124969883 | 2.04911351e-08/2.0654837e-08/0.00122106308/0.0305099949 |
| IMU | 20 | pseudo | 0/0/0.000120511569/0.00033204245 | 0.352630588 | 0.352630588 | 0/0/0.00214748397/0.00456261144 | 0/0/0.00747953456/0.0126896622 |

Typical probability-space epistemic remains near zero for the three accepted mappings. Normal/pseudo medians in nats are Camera 4.338551e-6 / 4.104429e-7, Seg 1.052007e-6 / 7.118817e-11, and IMU 3.823045e-11 / 0. Camera has no accepted probability-space UQ comparison at N=2 or N=10 because those mappings were invalid. Relative to N=10, Seg's normal median decreases from 3.124762e-6 while its pseudo median rises from 1.029926e-12 but remains tiny; IMU's normal median is similar to 3.719545e-11 and its pseudo median stays zero. These are descriptive comparisons of recomputed FIT/CAL compositions, without a new uncertainty threshold.

Near-zero medians do not mean every observation has zero epistemic. At N=20 normal/pseudo maxima are Camera 3.466041e-5 / 1.249715e-5, Seg 0.01842219 / 0.001518988, and IMU 0.002028813 / 0.0003320425 nats. Seg's normal tail maximum is larger than N=10's 0.00318025; IMU's normal maximum is smaller than N=10's 0.004924. Total and aleatoric medians remain close, as their small difference implies. Raw normal member-score standard-deviation medians are Camera 0.002478866, Seg 0.002056501 and IMU 0.0002524613; mapped probability standard-deviation medians are 0.001141419, 0.0001472142 and 2.065484e-8. IMU's normal probabilities concentrate near one, and the accepted sigmoid compresses typical raw disagreement strongly. This compares the existing metrics; it does not redefine uncertainty or establish physical safety. GNSS probability-space UQ is unavailable because its calibration remains invalid.

Historical N=2 medians come from accepted valid Seg/IMU outputs already saved in the accepted audit; their full probability-member distributions/intercept were not retained, so missing values are explicitly unavailable. N=10 probability disagreement is reconstructed read-only from saved accepted parameters/member scores, without refitting. Full historical comparison records are stored in comparison_results.json.

# N=20 Kaggle Handoff

Prepared handoff v2: `C:\Users\Aditya\AppData\Local\Temp\carla_train_expand20\kaggle_compact_train20_v1`, 35,571,358 bytes, 74,975 rows. No raw images/archives, graph examples or graph training. Includes frozen scenario/town/tick/15–5 partition metadata, compact features with explicit offsets, five bootstrap member scores per agent, validity/status, nullable calibrated prob_normal and canonical UQ, per-agent and observation constructed targets, pseudo parent/recipe/severity/seed/causal-window provenance, fitted parameters and cache/protocol/code/reuse hashes.

Invalid prob_normal/total/aleatoric/epistemic are NaN with an explicit per-agent validity mask, meaning scientific null. Prediction semantics remain P(target=normal) under the TRAIN-derived constructed clean-vs-pseudo task, without a physical-safety claim. All pseudo rows are CAL-only and cannot be repurposed as FIT graph-training examples.

# Storage / Timing

Acquisition including independent copy/replay/validation: 105.92 minutes. N=20 cache preprocessing/assembly: 80.63 minutes; analysis/export: 1.89 minutes. N20 staged prefix: 26.205 GB; newly extracted raw data: 14.627 GB; final cache total: 52.796 MB; handoff: 35.571 MB. All older prefixes/raw/cache/export artifacts remain separately preserved.

The measured N=20 preprocessing time includes clean extraction for only ten new scenarios plus newly needed CAL pseudo generation and deterministic assembly. It excludes the avoided repeat extraction of N=10 clean rows. Earlier N=10 preprocessing took 67.92 minutes; that is context rather than a measured counterfactual speedup.

Verified new clean feature blocks retained for reuse: 19.241 MB. The staging inventory records these blocks, raw/source manifests, logical view, final cache and handoff locations; original data and prior checkpoints remain separate.

New clean extraction overlapped acquisition: measured active clean-block intervals total 68.20 minutes; final assembly/CAL-pseudo wall interval is 12.43 minutes. The reported preprocessing sum is these measured intervals, not an additional serial elapsed delay. Per-scenario clean blocks contain no FIT/CAL assignment, models or pseudo rows; all final labels were assigned from the frozen twenty-scenario manifest.

# Tests / Core Integrity

Focused: 336 passed in 160.99s (0:02:40).

Exact requested broad suite: 505 passed, 14 warnings in 255.21s (0:04:15).

Broad baseline remains 505 tests; no production source or test files were changed. Existing N=2/N=10 reports are byte-hash preserved, original 3 GB and N=10 12.548 GB prefixes match their accepted hashes, all prior raw/artifact sizes/mtimes and cache/export byte hashes match the initial snapshot, generic/frozen source/tests remain unchanged, bare import cognix loads no CARLA adapters, and tracked source/test diff is empty.

No official TEST access, GAT/conformal changes, calibration/normality-score/feature/recipe/severity/threshold tuning, added regularization, slope caps, commits/pushes or destructive Git operations. Repository additions are confined to reports/carla_train_expand20_v1; the file inventory is in integrity_results.json. New temporary staging/raw/cache/export locations are explicitly recorded in the configuration and result manifests.

# Final Pre-GAT Modality Status

| Modality | Final status |
|---|---|
| Camera | calibrated-valid |
| Seg | calibrated-valid |
| GNSS | calibrated-invalid due structural separation (quasi_separated) |
| IMU | calibrated-valid |

This classification uses the frozen unchanged audit. It separates mathematical structural evidence from finite-optimization failure and keeps invalid probabilities/UQ unavailable. N=20 development data is frozen; no extra acquisition is proposed as a calibration remedy.

# Ready / Not Ready for GAT

Not ready to start GAT in this milestone. The N=20 development checkpoint is available for a separately preregistered methodology decision. Any invalid probability channels require an explicit scientific handling decision; graph sampling/targets/evaluation and FIT pseudo generation are also a later authorized stage. No graph construction, StandardGAT/EpistemicGAT training or conformal fitting occurred.

# Recommended Methodology Decision

Stop TRAIN expansion at N=20 and choose how to handle invalid mappings: GNSS. Preserve this dataset and the negative results. Additional scenario acquisition is not the automatic next step.

For the next milestone only: retain raw/member normality scores as uncalibrated statistics, or exclude invalid calibrated probability/UQ channels under a preregistered downstream comparison. If changing calibration, preregister the model/objective, validity criteria and evaluation before fitting; a regularized model would be a scientific protocol change, not a repair of the current unregularized MLE.

GNSS: distinguish loss of float64 exponential resolution from structural separation. A future score-representation study could retain pre-exponential distances/log-statistics, but recovering numeric resolution does not itself create a finite unregularized calibration MLE when the classes remain separated. A separately approved regularized/model alternative or exclusion decision would still need justification and validation.

None of these next-milestone options was implemented. Stop here: no N>20 acquisition, official TEST, GAT, conformal, scientific tuning, regularization or commits/pushes.

