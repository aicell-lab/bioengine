"""The 43 fields BBBC038 and BBBC039 share, so no client holds another's images.

Both collections draw on the BBBC022 Cell Painting screen and 43 fields are
pixel-identical between them. BBBC039's page mentions the overlap in prose and
publishes no list, so the list was recomputed from pixels: mean-removed,
L2-normalised 64x48 thumbnail correlation over the 97 BBBC038 fields whose
dimensions could match, threshold 0.99. Highest non-duplicate correlation 0.2294,
lowest duplicate 0.999923 -- nothing lands in between, so the cut is not a
judgement call. Cross-checked against BBBC038's metadata.xlsx Suppl. Table S1,
which lists project_id 'bbbc022' with num_images 43.

Derivation and full per-pair correlations:
bioengine-paper analysis/data/federated_consortium/bbbc038_bbbc039_overlap.json
(audit code: analysis/scripts/federated_consortium_fetch.py, audit-overlap).
"""

#: BBBC038 stage1_train directory names, dropped from the fluorescence client.
BBBC038_DUPLICATE_IDS = frozenset({
    "10ba6cbee4873b32d5626a118a339832ba2b15d8643f66dddcd7cb2ec80fbc28",
    "175dbb364bfefc9537931144861c9b6e08934df3992782c669c6fe4234319dfc",
    "1b518cd2ea84a389c267662840f3d902d0129fab27696215db2488de6d4316c5",
    "1c8b905c9519061d6d091e702b45274f4485c80dcf7fb1491e6b2723f5002180",
    "1f6b7cead15344593b32d5f2345fc26713dc74d9b31306c824209d67da401fd8",
    "2abc40c118bc7303592c8bb95a80361e27560854b8971ab34dcf91966575b1f2",
    "2cfa61bef6542dd359717e9131ce6f076c415a3bd7f48cb093b0d7f3b2ca785d",
    "3ebd2ab34ba86e515feb79ffdeb7fc303a074a98ba39949b905dbde3ff4b7ec0",
    "3f9fc8e63f87e8a56d3eaef7db26f1b6db874d19f12abd5a752821b78d47661e",
    "4596961c789d3b41916492918797724fe75128239fefc516c3ee75322b7926f0",
    "49edc2f7715100fb0390916e52b3fd11a921f02e59509dc987f67840a36250fc",
    "4d2cff9a0c8df3a7ef6100fda6f66e865a7670af6a18564767d8019b9ed2fd7b",
    "53ad09e4348767bece0165884bf40c10b72ae18444e3f414a850442f02385efc",
    "5f9d29d6388c700f35a3c29fa1b1ce0c1cba6667d05fdb70bd1e89004dcf71ed",
    "61dc249314d7b965eb4561ec739eab9b0f60af55c97b25ced8cb2a42a0be128e",
    "62057502c387145ed4f8f7f0d5e5bedcb72d3bcec15fa71cb0310dee32871461",
    "6b6d4e6ff52de473a4b6f8bd0f11ae22242d508cc4117ff38ec39cbb88088aaa",
    "6b72b61b80060a9e79a4747f9c5d5af135af9db466681c2d1086f784c7130699",
    "6fe2df6de1d962b90146c822bcefc84d0d3d6926fdfbacd3acdc9de830ee5622",
    "700afb1cd830a808e3c6125749612e5d23fd9f9726049a9e0c2061997514e1a7",
    "76faaed50ed6ea6814ac36199964b86fb09ba7f41a6f213bceaa80d625adc2e1",
    "876423522bdec1602917b94163a21e05fc7b692045219b7bc96cdaf638c33c25",
    "958114e5f37d5e1420b410bd716753b3e874b175f2b6958ebf1ec2bdf776e41f",
    "98a463483fe3a56deacc8bc00ab8aa62668bd40ad0c70bbe7deb10d3e4aeb0c0",
    "a02ec007ae8feddb758078b1dfb8010c26886fd3c8babdc308ead8b4a63acbdb",
    "a486f6ed4b8781e7883e433d06a83dd66db3e8b36d45b9976c4214820ee22629",
    "a65bbfc5673e8053b6ce49f39c79cf3a846fe5cc46dd93105f74fb07cf44606d",
    "a7a581e6760df4701941670e73d72533e3b0fbd7563488ad92772b41f7709710",
    "a891bbc89143bca7a717386144eb061ec2d599cba24681389bcb3a2fedb8ff8c",
    "aaa52100fafaa50877e777229cdf6cde7c422f145ff6719449b80631d9a3b0f6",
    "b1f23c4d27afed8af7b6b64793a3760bfea31b65f582d48aaa62d2b988ef2eac",
    "b76ff33ae9da28f9cd8bdce465d45f1eca399db3ffa83847535708e0d511fe38",
    "bb8ebf465c968a5f6f715de5d9e2e664afd1bcaa533e0e3352ecea1cc5b6fb0d",
    "bbfc4aab5645637680fa0ef00925eea733b93099f1944c0aea09b78af1d4eef2",
    "bde3727f3a9e8b2b58f383ebc762b2157eb50cdbff23e69b025418b43967556b",
    "c44ed955eb2e5c8d820b01477e122b32eff6dd475343e11229c33d8af3473b22",
    "cc88627344305b9a9b07f8bd042cb074c7a834c13de67ff4b24914ac68f07f6e",
    "d52958107d0b1f0288f50f346a833df3df485b92d5516cfcb536e73ab7adafd0",
    "dd54adb80393de7769b9853c0aa2ee9b240905d0e99c59d4ccd99401f327aa05",
    "e52960d31f8bddf85400259beb4521383f5ceface1080be3429f2f926cc9b5c2",
    "e5a6c5e01e6a4ef676a2d975374e995dd55792ea317a8e110bebc37da83a4ce8",
    "e7a3a7c99483c243742b6cfa74e81cd48f126dcef004016ad0151df6c16a6243",
    "ecb36c90cdd20245d89173c106f3c6a2d124d07bdea0ae202fb1efa49b0cd169",
})

#: BBBC039 image filenames, dropped from the bbbc039 client.
BBBC039_DUPLICATE_FILES = frozenset({
    "IXMtest_A18_s1_w1BFDF1C94-9C1F-4F5F-BBC1-05196333B1BF.tif",
    "IXMtest_B21_s4_w1521471E0-9BD7-492A-8739-9C782C2585B0.tif",
    "IXMtest_B21_s7_w1276972A2-1794-4A98-8E76-876A597C346B.tif",
    "IXMtest_C23_s2_w10533E965-3DC3-4556-9C26-EA8DD4F8DE61.tif",
    "IXMtest_D06_s5_w13C67AAA9-6E81-42DB-AC5F-7126602F3607.tif",
    "IXMtest_D10_s2_w1BB72A093-72AB-476C-9513-2FF43BFB2134.tif",
    "IXMtest_E19_s4_w129545707-4CD3-4498-AC27-E4AE24D0253C.tif",
    "IXMtest_F08_s1_w144C3056F-C4DD-4D39-A40F-4F4576A6DBD8.tif",
    "IXMtest_F08_s7_w117D78658-1BA7-495D-9EAA-E4FA155861AC.tif",
    "IXMtest_F22_s6_w1D6C33377-3947-4DAC-B62D-531C2565F150.tif",
    "IXMtest_G10_s3_w1C1257E17-1DBA-4619-B06E-D6DBB8A53088.tif",
    "IXMtest_G16_s3_w1B4690283-D75E-4DFB-92B0-29419E8292C6.tif",
    "IXMtest_G16_s6_w1C3F175E2-0AF5-401C-AC5C-8E128A237B22.tif",
    "IXMtest_H08_s8_w1418B93F8-515E-49A7-A9AE-CAF7A1589A38.tif",
    "IXMtest_H17_s1_w10A751E6C-5D06-4147-AB73-7FFAE0B57CC5.tif",
    "IXMtest_I04_s9_w16A5CC270-8B92-42EE-AA4A-855776F7D46B.tif",
    "IXMtest_I11_s6_w1B2DC04C7-2D7D-45C6-9DC2-66D8605FBE63.tif",
    "IXMtest_I15_s5_w11265571E-AA95-4E58-82CF-A75E3F240956.tif",
    "IXMtest_J02_s5_w1F53DDD35-C0B2-4E39-BA3B-9F66D289AB02.tif",
    "IXMtest_J02_s8_w1D9C198F9-ECF0-4EF7-848D-AC7782CD3C28.tif",
    "IXMtest_J15_s9_w1857B395F-AE7A-4319-BC26-3A041C0E31E9.tif",
    "IXMtest_J16_s8_w1AD7BC3EB-6A29-4A53-AA59-E39653839B0D.tif",
    "IXMtest_J17_s1_w19E7BCD2F-6D7E-4B39-9154-F93E4D539183.tif",
    "IXMtest_J20_s1_w1EEE65E52-7AD8-47C7-A286-6E84C5D77953.tif",
    "IXMtest_K05_s7_w1816D5E2A-B9D8-48DA-9FE6-CBC35FE38C8D.tif",
    "IXMtest_K06_s9_w1787DFDA5-E2CE-40DB-9430-CC3766599BCC.tif",
    "IXMtest_K12_s7_w12A7857A5-3C92-4A08-8E81-2CA8A99F67AE.tif",
    "IXMtest_K18_s8_w11FC8D3EA-E30D-46B8-A45E-0399327AA8AC.tif",
    "IXMtest_L01_s3_w1E7E0D198-5FB4-4E10-A27C-C46463DA9E06.tif",
    "IXMtest_L03_s6_w1BE79472D-5E2C-422F-A16E-7AC0691C0FD8.tif",
    "IXMtest_L21_s5_w122478CD2-80DC-4B4E-9BC8-A6F6239F4103.tif",
    "IXMtest_M04_s3_w1A599DF67-1E7F-4A09-84FD-0080767A735C.tif",
    "IXMtest_M20_s3_w15C73A7C7-F81B-4583-AB8F-0A64336AF070.tif",
    "IXMtest_N07_s5_w1D30ED7AB-503E-479D-B5BC-B66472568DE2.tif",
    "IXMtest_N08_s2_w159DEEDD5-2882-4F9E-9048-F5BD2B918651.tif",
    "IXMtest_N12_s9_w1C98EC3CC-2F8E-4761-A21A-EBF14469AD40.tif",
    "IXMtest_N21_s1_w14F264B20-62D6-47EC-9BCD-B9416AB302D9.tif",
    "IXMtest_O10_s8_w18F4DB020-BFB7-4F13-B99C-C39F8E54F85D.tif",
    "IXMtest_O16_s3_w129E61E12-E398-496E-9408-092D2D70E74B.tif",
    "IXMtest_O18_s7_w19C30A212-58D3-4030-AA4F-B0C4482F1F8A.tif",
    "IXMtest_P10_s7_w1F78192AF-7D11-4D41-80F3-8CD6DB05AB57.tif",
    "IXMtest_P13_s6_w1270AEE06-D31F-421C-8147-13644F4C5B94.tif",
    "IXMtest_P23_s7_w13B627CB9-6C57-4049-AAD7-6468A051DD24.tif",
})

#: Three BBBC039 masks hold no annotated nucleus. None is one of the 43
#: duplicates, so all three survive de-duplication. An all-background target
#: makes a Dice loss degenerate (0/0), so they are dropped here by name rather
#: than left to whatever the loss does with a zero denominator. Matched on the
#: well_site prefix; the rest of each filename is a per-file UUID.
BBBC039_EMPTY_MASKS = ("IXMtest_F13_s7_", "IXMtest_L01_s2_", "IXMtest_L10_s6_")
