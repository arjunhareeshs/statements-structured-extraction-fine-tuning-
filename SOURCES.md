# Data provenance

All 129 cases in `data/*.jsonl` are real, named grassroots innovators. Facts
were gathered via web search (this session, July 2026) from the sources
below. Paragraph text and `problem`/`solution` labels in the dataset are
**original paraphrases written from these facts, not copied text** (kept
under standard paraphrase/copyright limits — no source is quoted verbatim).

## Original 37 cases

| case_id | Primary source(s) |
|---|---|
| remya_jose | Honey Bee Network coverage (nalakagunawardene.com, 2011); "From Sink to Source" (Honey Bee Network monograph) |
| appachan | nalakagunawardene.com, "3 Idiots and Honey Bee Network..." (2011) |
| saidullah | same as above |
| dhanjibhai | same as above |
| jagani_sprayer, jayanti_patel, dodhi_pathak, jagtap_spray, kamruddin_bike, rathore_pump, gayen_pump, kanak_das | SlideShare deck "Honey bee network" (compiled Honey Bee Network bicycle-innovation list) |
| marvaniya_carrot | en.wikipedia.org/wiki/Vallabhbhai_Marvaniya; nif.org.in/news/485 (Padma Shri 2019 announcement) |
| cv_raju_toys | en.wikipedia.org/wiki/C._V._Raju |
| dipak_sardar, suthar_groundnut, maurya_guava, aniyamma_cashew, chopade_embossing, majhi_transplanter | nif.org.in, 11th National Grassroots Innovation Award Book (PDF, April 2023) |
| yusuf_groundnut_digger | nif.org.in/innovation/groundnut-digger/745 |
| parikh_cauliflower, pachar_carrot, kumawat_thresher, ola_heat_recovery, dahiya_gasifier, sundaram_verma | nif.org.in/news/535 ("Journey of an innovator to an innovation influencer") |
| khass_planter | nif.org.in/news/339 ("Rural Innovations: Transforming Lives") |
| mohanbhai_thresher | nif.org.in/innovation/mobile-groundnut-thresher-cum-collector/585 |
| nilesh_pankaj_thresher | nif.org.in/innovation/mobile-groundnut-thresher/1096 |
| nadakattin_tamarind, nadakattin_drill | en.wikipedia.org/wiki/Abdul_Khader_Nadakattin; devdiscourse.com (Padma Shri 2022 announcement) |
| bharali_deseeder | en.wikipedia.org/wiki/Uddhab_Bharali |
| hariman_apple | nif.org.in homepage news item (Padma Shri, HRMN-99 apple variety) |
| manipur_poultry_herbal, kundu_herbal | ecoideaz.com, "NIF's Award Winning Grassroots Innovators from Rural India"; nif.org.in/innovation/herbal_formulation_kamaal_505/364 |
| prajapati_mitticool | Widely reported (Better India, multiple mainstream Indian news outlets); not re-verified by a fresh search this session — flagged below |

## 92 additional cases (added to reach ~400 train rows)

All sourced from NIF's own National Biennial Award Books, each of which
profiles several dozen awardees per edition with name, place, innovation
title, and a "problem addressed" write-up in NIF's own words (paraphrased
here, not copied).

| case_id | Primary source |
|---|---|
| indrajit_khass_stump, karekar_turmeric, shishpal_herbal, biju_coconut, singh_cauliflower, vishnu_mastitis, karappan_handloom, shine_clove, jalendra_ironbar, shafi_walnut, imkong_pineapple, chhuanmawia_laddu, yanglem_taro, suchil_soilcrusher, aminuddin_tarpaulin, rajesh_polanga, shinde_silkworm, lakkad_sorghum, aji_pelletisation, prusty_strawcutter, binoy_pepper, dhedhi_cultivator, dinesh_channel, mohan_cuphanger, selichum_cardamom, budheswar_coconut, harkumar_juice, gobin_arecanut, chinnakannu_moulding, punamchand_onion, mousumi_rice, anand_chilli, akhmabhai_anestrus, baltej_flower, nisar_standingbar, bajrang_wheat, hom_leafcurl, asim_kangri, sayen_lpg, sumit_bamboo, tsering_seabuckthorn, hema_stretcher, arushi_bed, mauwang_lifejacket | nif.org.in/upload/11th-Final-Award-Book-April-2023.pdf (11th National Biennial Grassroots Innovations and Outstanding Traditional Knowledge Awards, 2023) |
| raghuvanshi_seeds, shyambir_transplanter, sandip_onion, anang_goggles, periyasami_coccidiosis, saravanamuthu_bed, sanjeev_cauliflower, vilat_anestrus, maharaj_tillage, prafulla_weft, ramprasad_weft, suresh_arecanut, jitabhai_bean, laxmanbhai_bloat, keisham_cauliflower, dattatraya_grapes, roshan_pestcontrol, jyotsna_pestcontrol, dayaram_tamarind, kacharu_bloat, vijayram_bloat, imna_optable, ravi_jacquard, sekar_weaving, durlov_teadryer, suren_coolingbed, tirupathi_pole, arjunbhai_crematorium, durgadevi_crutch, babasaheb_onion, roymathew_nutmeg, helendro_chilli, vithal_grapes, kishansingh_pestcontrol, divya_pestcontrol, laxmiben_bloat, shamalbhai_mastitis, jhabarmal_harvester, lakhanlal_sunflower, saruj_reeling, jagtap_sandalwood, sapan_swing, mohan_cowdung, towseef_bukhari, lanu_loadcontroller, sourav_lac, zufa_namda, evthomas_nutmeg | nif.org.in/upload/10th-National-Award-Book-2019-new.pdf (10th National Grassroots Innovation and Outstanding Traditional Knowledge Awards, 2019) |

## Honesty flags (please read before treating this as production data)

- **Auto-generated, not hand-verified.** Every `problem`/`solution` label was
  written by Claude (this session) directly from the sourced facts above,
  as the task brief explicitly permits ("bootstrap labels with an API LLM
  ... be honest about what was human-verified vs auto-generated"). **None
  of these labels have been independently hand-verified by a second human
  reviewer.** Before using this as a trusted benchmark, spot-check a
  random ~15-20% sample against the source URLs above.
- **`prajapati_mitticool`** relies on well-known, widely corroborated public
  facts about Mansukhbhai Prajapati's Mitticool clay refrigerator, but was
  not re-confirmed against a specific URL in this session — treat it as
  lower-confidence than the other 36 rows and verify independently if it
  matters for your use case.
- **Paraphrase augmentation.** Each of the 129 real cases was expanded into
  4-5 differently-worded paragraphs describing the *same* underlying story
  (see `build_dataset.py`, functions `t1`-`t5`). This is a legitimate,
  disclosed data-augmentation step to reach a usable row count from a
  realistic amount of independently-sourced source material in one
  session — it is not 518 independent stories, it is 129. The train/val/test
  split is done by **case**, not by row, specifically so this augmentation
  cannot leak a paraphrase of a test-set story into training (see
  `build_dataset.py`'s leakage assertion).
- **Current split (regenerated after the 92-case expansion):** 129 cases →
  414 train rows (103 cases) / 52 val rows (13 cases) / 52 test rows (13
  cases). Still modest for judging generalization, but a meaningful step up
  from the original 37-case / 12-test-row split. If you extend this project
  further, the same NIF award-book approach (other biennial editions, listed
  in the main README) is the fastest path to more independent cases.
