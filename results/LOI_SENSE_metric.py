import os
import os.path as osp
import json
import numpy as np
import torch
base_dir = '/data/GJY/ht/CVPR2022/dataset/SENSE'
def LOI_metrics_in_sense():
    cross_line=[59.883488224601024, 130.2718122601509, 22.186551724185847, 36.969887905041105, 23.258880632463843, 44.26441172834893, 39.22899675931694, 118.26700779050589, 28.41176374574394, 31.080895460691863, 225.94703990221024, 32.82147675240412, 127.4519899688894, 4.352750147198094, 11.641417545364789, 56.93439202217269, 20.7099011287537, 27.377433187964925, 34.01371757276502, 64.00487532333591, 78.81866610534053, 25.30856114084064, 21.409252693667895, 24.555483096714514, 24.748112513823344, 10.429442754191314, 36.0664805631659, 62.393183347131526, 52.71877195353136, 185.98395701067057, 25.520133665904496, 35.92141709667703, 35.413785373851056, 37.52936681883784, 22.99515816069203, 34.21141566169899, 20.690827254452415, 21.3110946378074, 18.72128474596275, 60.94368629574092, 17.949290528855727, 88.64402703531232, 93.06084786914289, 16.283471058620194, 21.01408856402395, 25.754371689959953, 104.47728612684296, 109.3703624010086, 106.05313750787172, 39.85242473389553, 48.1321870405809, 17.236021038414037, 69.0967676952157, 50.22762099386637, 51.60301931888853, 20.694838109005104, 153.5907784635201, 27.338370766031403, 31.747310932720666, 41.76116591139362, 46.7754753967165, 61.14391249478001, 36.653548788279295, 6.753133285375952, 22.47836683654168, 21.77090639117705, 106.84877285582479, 92.39074369217269, 18.019730334530607, 7.147473837541838, 55.94333603233224, 139.677742655389, 59.67927178248647, 17.45766474559059, 40.9461781293553, 10.742174956965755, 31.942700325962505, 10.189139181461428, 18.073547753538833, 46.03903707070276, 11.880232329422142, 40.09705050623188, 18.598055237985136, 131.2656190288253, 35.29435838894233, 34.61036969477482, 61.679560874919844, 28.286659178561763, 146.1206465146388, 10.930097709277106, 21.058210979433625, 37.247306169163494, 21.710714887038193, 19.458648529640794, 36.935653935292976, 86.37092534755357, 31.839857131516055, 176.61565245501697, 210.56136168539524, 9.587812577261502, 36.506314970006315, 44.27753110131016, 29.58407960186014, 223.35476322472095, 68.16879293222519, 24.719829507319957, 22.40062587785542, 83.07005946338177, 157.8038376495242, 54.303807156430366, 29.885924529150543, 37.93216871493132, 36.71998666901527, 120.92748659197241, 44.475013826740906, 86.89708166760829, 31.28989564492349, 47.9618204019589, 23.941276541209845, 35.141972786412225, 23.81997632746004, 36.14212249066324, 57.46973707580037, 23.313777665958952, 10.024855962536094, 39.12520546273481, 37.26702824170752, 19.723607776878097, 81.5845951985284, 56.69408562683384, 37.21087729464131, 129.7884387075901, 21.06809635594857, 31.398001569206826, 51.966776381679665, 23.729230998725043, 185.92321610450745, 130.43264532089233, 30.28968345011924, 49.596310366439866, 47.47501884501253, 37.42925588321509, 42.807190975563145, 37.46559424584302, 35.101448738075305, 77.17056874092549, 120.870865306817, 60.47341917056474, 56.778412656101864, 71.05525756639145, 35.23215961691312, 182.78791970852762, 14.326641353089144, 53.57397911486623, 44.98188718830602, 13.448171493746088, 21.65039175118623, 42.6087092228845, 25.383385313128443, 73.4346116805682, 34.0140626566257, 38.6493722978048, 121.50496331229442, 43.66745995638121, 14.917694212390984, 7.149453309022192, 26.08553043100983, 55.37848237294111, 20.685752979456083, 23.895146843909963, 27.404360450394577, 43.08535429407493, 18.683511120355888, 18.016959482304628, 22.699059587859665, 41.1771790292114, 67.21999072666267, 87.48604187271485, 17.073765660797108, 50.01258381097068, 40.991837323964546, 25.319426787197187, 16.067126560250472, 14.269390169531107, 21.840840490486332, 38.510274361646225, 5.028799353263821, 25.29516133720972, 93.91357424219314, 109.73449916236859, 16.631580541461517, 79.08851487212814, 50.13711871560372, 43.8376854861217, 175.41213149949908, 15.765760296701501, 39.84424976014634, 51.12212937422737, 36.69965171539057, 71.03823228732472, 23.144847373700486, 4.1770715242555525, 63.77370557456834, 65.94239916918013, 23.606910055596018, 39.882952491985634, 8.003772563707798, 34.885039431102996, 71.1635716734454, 10.710768866729495, 59.344121536596504, 45.516102213202885, 17.06016651707614, 25.9010779770025, 300.52123618125916, 7.947268502547217, 50.71802872344489, 86.66593653300879, 32.32803700925433, 12.729242554065422, 26.20703925556245, 12.364015790304592, 17.572867198382482, 14.585915625426424, 37.26532668903016, 62.92536462932526, 37.29714956067528, 33.31797270430252, 31.194981854576326, 12.162939427266394, 72.88409650349149, 23.349309819425592, 14.979270775654186, 51.395475158999034, 39.4866532303522, 30.98420115290594, 11.862033709346179, 38.86166944156503, 50.94480476375293, 32.96826659395035, 11.552053533810636, 60.00151374563575, 27.82052986314875, 30.80500936739554, 70.69998336397111, 49.279962293745484, 34.91348677261499, 22.66675749373303, 35.60396577743694, 29.527725639419884, 13.596589878015038, 22.91391229191504, 34.49798774668199, 38.73985504316806]
    density_sample = [329.2447090148926, 255.91525268554688, 93.05430698394775, 181.0734748840332, 120.83247756958008, 155.84905624389648,
                   195.4491672515869, 653.9289779663086, 149.2770538330078, 156.04193115234375, 670.7795028686523, 155.37398147583008,
                   720.8888778686523, 29.27963948249817, 128.32279682159424, 223.1608009338379, 49.02521753311157, 170.41716957092285,
                   210.95069313049316, 338.8018112182617, 408.48405838012695, 169.83823585510254, 167.70843315124512, 151.7740135192871,
                   181.3070182800293, 65.80084419250488, 302.44388580322266, 260.9303913116455, 356.26223373413086, 846.9485702514648,
                   148.07650184631348, 113.18601036071777, 247.44953536987305, 172.56501007080078, 165.59934043884277, 232.7839183807373,
                   94.14818000793457, 55.68123757839203, 98.98388481140137, 312.15608978271484, 92.70579719543457, 514.4846267700195,
                   350.59794998168945, 97.92344951629639, 94.06393814086914, 191.66877365112305, 501.09375, 189.15494537353516, 730.560432434082,
                   239.3647975921631, 331.0234489440918, 80.55887508392334, 425.6973114013672, 277.1560592651367, 255.08352661132812,
                   131.31836223602295, 679.5399894714355, 129.62917518615723, 162.8681755065918, 173.73694610595703, 213.2879123687744,
                   272.8165111541748, 185.6861457824707, 80.80370712280273, 113.78744506835938, 103.60079193115234, 477.38951110839844,
                   503.0850372314453, 99.07415103912354, 66.00294637680054, 336.2936019897461, 746.2285537719727, 227.6953010559082,
                   81.15500736236572, 168.15549087524414, 73.67917156219482, 141.3722915649414, 56.25684928894043, 83.25054836273193,
                   124.44530868530273, 56.90091943740845, 229.95313453674316, 75.79217433929443, 568.3032836914062, 216.9452075958252,
                   170.07469177246094, 398.87407302856445, 278.6100959777832, 867.331657409668, 70.17723059654236, 177.72316932678223,
                   207.89899063110352, 132.17094326019287, 107.2018871307373, 163.44209480285645, 230.53589344024658, 187.07168579101562,
                   1032.2536315917969, 603.0247344970703, 21.711125373840332, 236.53734588623047, 253.3560676574707, 50.891855239868164,
                   948.010612487793, 246.92536735534668, 147.6364974975586, 154.90376472473145, 278.6030578613281, 765.3510665893555,
                   329.2731742858887, 159.07469940185547, 226.9019012451172, 174.57555389404297, 615.8282318115234, 133.86121559143066,
                   390.96221923828125, 210.80270957946777, 161.64145278930664, 213.0253620147705, 137.8933277130127, 159.13882446289062,
                   200.70980644226074, 196.69668579101562, 133.99272537231445, 58.485106468200684, 255.4698028564453, 208.08446502685547,
                   98.5499439239502, 517.9687957763672, 247.07362747192383, 257.3138904571533, 328.66653060913086, 131.32260990142822,
                   147.67138290405273, 201.82437896728516, 109.86320495605469, 497.5025291442871, 272.55457305908203, 187.750581741333,
                   289.36083602905273, 283.34507751464844, 244.40475845336914, 218.27612113952637, 187.4944190979004, 191.84341049194336,
                   433.79409408569336, 605.7817764282227, 288.9931945800781, 175.87884902954102, 429.34496307373047, 152.4539279937744,
                   537.7921295166016, 72.17854595184326, 356.5210952758789, 294.6801452636719, 69.58928871154785, 125.55422878265381,
                   206.23386573791504, 115.0451831817627, 344.48927307128906, 131.23259830474854, 235.55986404418945, 806.1847839355469,
                   236.1481819152832, 63.441771030426025, 38.93693161010742, 163.9031753540039, 293.6746711730957, 124.82248306274414,
                   79.10861206054688, 162.12722206115723, 146.83791160583496, 101.46988868713379, 90.99212265014648, 111.31876850128174,
                   108.62157249450684, 377.6471252441406, 477.4294662475586, 59.906328201293945, 230.98709869384766, 203.3636474609375,
                   151.50853729248047, 68.91777563095093, 74.24030208587646, 124.32095241546631, 196.9592399597168, 37.91118109226227,
                   161.32350730895996, 409.08020401000977, 763.3636322021484, 72.0870418548584, 415.8499641418457, 276.55988121032715,
                   218.30863189697266, 829.9740219116211, 112.68836975097656, 251.23281288146973, 246.98335456848145, 200.48199081420898,
                   343.2992248535156, 155.83619689941406, 31.409766912460327, 277.8610610961914, 419.78992080688477, 129.00805854797363,
                   229.75195693969727, 44.17517852783203, 154.20663452148438, 410.2150230407715, 90.44812870025635, 307.92285537719727,
                   328.01897048950195, 83.45688247680664, 130.54365348815918, 1390.4736328125, 25.418426990509033, 359.72925186157227,
                   509.7667236328125, 143.6260929107666, 83.39159393310547, 144.45436096191406, 61.738046646118164, 71.6042218208313,
                   77.98232460021973, 200.9752597808838, 444.5205383300781, 167.1799201965332, 194.12027168273926, 165.4094409942627,
                   78.90681266784668, 406.80435943603516, 102.1224946975708, 103.30379104614258, 307.5182113647461, 146.7274990081787,
                   137.6467113494873, 68.7066559791565, 413.1406440734863, 506.50048065185547, 173.0795955657959, 101.74421977996826,
                   388.3800277709961, 162.97664642333984, 128.6250820159912, 239.17142868041992, 287.6471939086914, 214.09355354309082,
                   120.22795104980469, 190.93337440490723, 177.59774017333984, 81.95251178741455, 109.90049076080322, 191.6724510192871,
                   134.4361686706543]
    density_max=[54.66169357299805, 137.0662384033203, 16.068391799926758, 30.9769344329834, 21.338228225708008, 43.523094177246094,
                    36.353946685791016, 104.12767028808594, 26.01197052001953, 26.96659278869629, 106.85093688964844, 26.744001388549805,
                    111.67567443847656, 6.456052303314209, 26.926382064819336, 42.33304214477539, 9.327913284301758, 30.18781280517578,
                    37.577903747558594, 56.68721008300781, 64.96417236328125, 29.757465362548828, 29.95824432373047, 26.74517059326172,
                    30.52520751953125, 12.169918060302734, 53.65992736816406, 49.565582275390625, 62.68394088745117, 71.64127349853516,
                    24.853635787963867, 21.691267013549805, 38.08953857421875, 30.998552322387695, 30.29684066772461, 39.96257019042969,
                    18.94898223876953, 19.17557144165039, 17.16815185546875, 51.753196716308594, 16.224348068237305, 82.83113098144531,
                    57.619422912597656, 17.003400802612305, 18.921215057373047, 34.20343017578125, 81.88605499267578, 102.44983673095703,
                    115.00328063964844, 40.14585876464844, 54.473304748535156, 17.912769317626953, 68.29122924804688, 49.94713592529297,
                    43.318634033203125, 26.768993377685547, 62.83100128173828, 21.83800506591797, 28.63059425354004, 29.627880096435547,
                    38.44188690185547, 56.86794662475586, 29.51821517944336, 17.800609588623047, 19.508359909057617, 19.43529510498047,
                    76.8759994506836, 79.94617462158203, 18.87087631225586, 14.261246681213379, 55.81650161743164, 130.6891632080078,
                    37.49557876586914, 14.344047546386719, 31.84566879272461, 13.583511352539062, 26.65854835510254, 10.38752555847168,
                    16.560821533203125, 20.37192153930664, 12.166696548461914, 40.93467330932617, 16.01966094970703, 94.11332702636719,
                    35.266815185546875, 28.786348342895508, 66.43228149414062, 24.369327545166016, 134.415283203125, 19.54155731201172,
                    32.27302932739258, 36.122032165527344, 27.371841430664062, 19.944631576538086, 33.26054382324219, 49.41266632080078,
                    35.94380569458008, 153.7733154296875, 98.1309814453125, 9.008898735046387, 36.76648712158203, 41.6186408996582,
                    26.830615997314453, 152.50558471679688, 42.30065155029297, 25.413280487060547, 28.989429473876953, 61.443504333496094,
                    117.47796630859375, 55.29071807861328, 27.052860260009766, 40.830055236816406, 31.658645629882812, 101.48677825927734,
                    26.46095085144043, 72.50664520263672, 35.35853576660156, 25.488605499267578, 38.66465377807617, 23.71792221069336,
                    27.38129425048828, 34.74257278442383, 35.629486083984375, 22.415475845336914, 14.049745559692383, 42.88771057128906,
                    34.70915222167969, 18.341976165771484, 80.36051177978516, 42.679080963134766, 43.90382766723633, 56.21989059448242,
                    24.79953384399414, 27.45168685913086, 37.2308235168457, 20.422889709472656, 81.40318298339844, 44.790924072265625,
                    29.880537033081055, 49.858219146728516, 49.1226806640625, 39.565818786621094, 38.54889678955078, 32.032936096191406,
                    33.80495071411133, 76.99103546142578, 97.64599609375, 47.241294860839844, 39.871490478515625, 76.76459503173828,
                    35.04552459716797, 64.99618530273438, 15.624757766723633, 58.02946090698242, 46.514129638671875, 11.763120651245117,
                    23.453350067138672, 34.00738525390625, 24.03924560546875, 53.992828369140625, 28.05129623413086, 40.55502700805664,
                    120.85169219970703, 39.023284912109375, 13.598579406738281, 12.278825759887695, 34.456764221191406, 52.33754348754883,
                    26.529075622558594, 15.570074081420898, 31.366313934326172, 23.931488037109375, 19.204082489013672, 20.02055549621582,
                    22.17809295654297, 19.88871955871582, 64.07493591308594, 76.97769165039062, 24.192752838134766, 40.978271484375,
                    37.375144958496094, 25.863513946533203, 16.696277618408203, 12.505027770996094, 21.469526290893555, 34.985408782958984,
                    9.487812042236328, 28.70437240600586, 68.64350891113281, 127.36717224121094, 14.1132173538208, 65.52947998046875,
                    48.912620544433594, 39.19871139526367, 131.52496337890625, 20.570934295654297, 43.238487243652344, 46.80751037597656,
                    35.77086639404297, 60.11964416503906, 25.830337524414062, 6.880366802215576, 45.22911071777344, 78.66773986816406,
                    24.081071853637695, 40.05316925048828, 9.325295448303223, 26.78860855102539, 79.85025024414062, 21.83191680908203,
                    58.477664947509766, 56.548118591308594, 13.685639381408691, 33.72235870361328, 211.89047241210938, 9.851618766784668,
                    62.343849182128906, 79.49717712402344, 24.53809356689453, 16.12444305419922, 26.50648307800293, 11.378901481628418,
                    14.498477935791016, 14.122715950012207, 40.951663970947266, 73.16452026367188, 30.0306453704834, 34.52653503417969,
                    28.053110122680664, 14.847644805908203, 65.91864776611328, 19.635135650634766, 20.986738204956055, 50.38282012939453,
                    28.292640686035156, 25.711849212646484, 14.040419578552246, 102.1163330078125, 48.74378204345703, 29.865028381347656,
                    21.547157287597656, 63.69385528564453, 33.06465148925781, 24.64487648010254, 39.8432731628418, 48.52802276611328,
                    34.04418182373047, 22.703792572021484, 34.84085464477539, 29.802276611328125, 15.714008331298828, 19.739463806152344,
                    30.41762351989746, 24.20845603942871]
    cross_line = density_max
    print(len(cross_line))
    with open(osp.join(base_dir, 'scene_label.txt'), 'r') as f:
        lines = f.readlines()
    scene_label = {}
    for line in lines:
        line = line.rstrip().split(' ')
        scene_label.update({line[0]: [int(i) for i in line[1:]]})

    scenes_pred_dict = {'all':[], 'in':[], 'out':[], 'day':[],'night':[], 'scenic0':[], 'scenic1':[],'scenic2':[],
                      'scenic3':[],'scenic4':[],'scenic5':[], 'density0':[],'density1':[],'density2':[], 'density3':[],'density4':[] }
    scenes_gt_dict =  {'all':[], 'in':[], 'out':[], 'day':[],'night':[], 'scenic0':[], 'scenic1':[],'scenic2':[],
                      'scenic3':[],'scenic4':[],'scenic5':[], 'density0':[],'density1':[],'density2':[], 'density3':[],'density4':[] }
    time_dict = {'all': [], 'in': [], 'out': [], 'day': [], 'night': [], 'scenic0': [], 'scenic1': [], 'scenic2': [],
                      'scenic3': [], 'scenic4': [], 'scenic5': [], 'density0': [], 'density1': [], 'density2': [], 'density3': [], 'density4': []}
    with open(os.path.join(base_dir, 'test.txt'), 'r') as txt:
        scene_names = txt.readlines()
        scene_names = [i.strip() for i in scene_names]


    with open('../datasets/dataset_prepare/info.json','r') as f:
        gt_info = json.load(f)


    for i, (scene_name, pred_dict) in enumerate(zip(scene_names, cross_line)):

        time = len(os.listdir(osp.join(base_dir, 'video_ori', scene_name)))
        gt_dict = gt_info[scene_name]
        print(scene_name, pred_dict, gt_dict)
        scenes_pred_dict['all'].append(pred_dict)
        scenes_gt_dict['all'].append(gt_dict)
        time_dict['all'].append(time)
        scene_l = scene_label[scene_name]
        if scene_l[0] == 0: scenes_pred_dict['in'].append(pred_dict);  scenes_gt_dict['in'].append(gt_dict); time_dict['in'].append(time)
        if scene_l[0] == 1: scenes_pred_dict['out'].append(pred_dict);  scenes_gt_dict['out'].append(gt_dict); time_dict['out'].append(time)
        if scene_l[1] == 0: scenes_pred_dict['day'].append(pred_dict);  scenes_gt_dict['day'].append(gt_dict); time_dict['day'].append(time)
        if scene_l[1] == 1: scenes_pred_dict['night'].append(pred_dict);  scenes_gt_dict['night'].append(gt_dict); time_dict['night'].append(time)
        if scene_l[2] == 0: scenes_pred_dict['scenic0'].append(pred_dict);  scenes_gt_dict['scenic0'].append(gt_dict); time_dict['scenic0'].append(time)
        if scene_l[2] == 1: scenes_pred_dict['scenic1'].append(pred_dict);  scenes_gt_dict['scenic1'].append(gt_dict); time_dict['scenic1'].append(time)
        if scene_l[2] == 2: scenes_pred_dict['scenic2'].append(pred_dict);  scenes_gt_dict['scenic2'].append(gt_dict); time_dict['scenic2'].append(time)
        if scene_l[2] == 3: scenes_pred_dict['scenic3'].append(pred_dict);  scenes_gt_dict['scenic3'].append(gt_dict); time_dict['scenic3'].append(time)
        if scene_l[2] == 4: scenes_pred_dict['scenic4'].append(pred_dict);  scenes_gt_dict['scenic4'].append(gt_dict); time_dict['scenic4'].append(time)
        if scene_l[2] == 5: scenes_pred_dict['scenic5'].append(pred_dict);  scenes_gt_dict['scenic5'].append(gt_dict); time_dict['scenic5'].append(time)
        if scene_l[3] == 0: scenes_pred_dict['density0'].append(pred_dict);  scenes_gt_dict['density0'].append(gt_dict); time_dict['density0'].append(time)
        if scene_l[3] == 1: scenes_pred_dict['density1'].append(pred_dict);  scenes_gt_dict['density1'].append(gt_dict); time_dict['density1'].append(time)
        if scene_l[3] == 2: scenes_pred_dict['density2'].append(pred_dict);  scenes_gt_dict['density2'].append(gt_dict); time_dict['density2'].append(time)
        if scene_l[3] == 3: scenes_pred_dict['density3'].append(pred_dict);  scenes_gt_dict['density3'].append(gt_dict); time_dict['density3'].append(time)
        if scene_l[3] == 4: scenes_pred_dict['density4'].append(pred_dict);  scenes_gt_dict['density4'].append(gt_dict); time_dict['density4'].append(time)


    for key in scenes_pred_dict.keys():

        pre_crowdflow_cnt = torch.tensor(scenes_pred_dict[key]).float()
        gt_crowdflow_cnt = torch.tensor(scenes_gt_dict[key]).float()
        time =torch.tensor(time_dict[key]).float()


        MAE = torch.mean(torch.abs(pre_crowdflow_cnt -gt_crowdflow_cnt))
        MSE = torch.mean((pre_crowdflow_cnt- gt_crowdflow_cnt) ** 2).sqrt()
        WRAE = torch.sum(torch.abs(pre_crowdflow_cnt -gt_crowdflow_cnt) /gt_crowdflow_cnt* (time/time.sum())) * 100
        print('=' * 20, key, '=' * 20)
        print('MAE: %.2f, MSE: %.2f  WRAE: %.2f ' % (MAE.data, MSE.data, WRAE.data))


from train import compute_metrics_all_scenes
from collections import  defaultdict
import numpy as np
import cv2
from PIL import  Image
def tracking_to_crowdflow():

    dataset_root = '/media/D/GJY/ht/CVPR2022/dataset/SENSE/'
    tracking_result_root = './sense_tracking_results'

    scenes = os.listdir(tracking_result_root)

    scenes_pred_dict = []
    scenes_gt_dict = []
    all_sum  = []
    with open(osp.join(base_dir, 'scene_label.txt'), 'r') as f:
        lines = f.readlines()
    scene_label = {}
    for line in lines:
        line = line.rstrip().split(' ')
        scene_label.update({line[0]: [int(i) for i in line[1:]]})



    scenes_pred_dict = {'all':[], 'in':[], 'out':[], 'day':[],'night':[], 'scenic0':[], 'scenic1':[],'scenic2':[],
                      'scenic3':[],'scenic4':[],'scenic5':[], 'density0':[],'density1':[],'density2':[], 'density3':[],'density4':[] }
    scenes_gt_dict =  {'all':[], 'in':[], 'out':[], 'day':[],'night':[], 'scenic0':[], 'scenic1':[],'scenic2':[],
                      'scenic3':[],'scenic4':[],'scenic5':[], 'density0':[],'density1':[],'density2':[], 'density3':[],'density4':[] }

    for _, i in enumerate(scenes,0):
        # if _>0:
        #     break
        scene_name = i.split('.')[0]
        pred = defaultdict(list)
        gts = defaultdict(list)
        detect_points = defaultdict(list)
        path = os.path.join(tracking_result_root,i)
        id_list  = []
        with open(path, 'r') as f:
            lines = f.readlines()
            for vi, line in enumerate(lines, 0):
                line = line.strip().split(',')
                img_id = int(line[0])
                tmp_id = int(line[1])
                detect_points[img_id].append([float(line[2]) + float(line[4]) / 2, float(line[3]) + float(line[5]) / 2])
                pred[img_id].append(tmp_id)
                id_list.append(tmp_id)
        with open(osp.join(dataset_root, 'label_list_all', i), 'r') as f:
            lines = f.readlines()
            for lin in lines:
                lin_list = [i for i in lin.rstrip().split(' ')]
                ind = int(lin_list[0].split('.')[0])
                print(i, ind)
                lin_list = [float(i) for i in lin_list[3:] if i != '']
                assert len(lin_list) % 7 == 0
                box_and_point = torch.tensor(lin_list).view(-1, 7).contiguous()
                gts[ind] = (box_and_point[:, 6]).numpy()

        id = set(id_list)
        all_sum.append(len(id))
        # print(all_sum, sum(all_sum[:4]), sum(all_sum[4:]))

        pred_dict = {'id': i, 'time': len(pred.keys()), 'first_frame': 0, 'inflow': [], 'outflow': []}
        gt_dict = {'id': i, 'time': len(pred.keys()), 'first_frame': 0, 'inflow': [], 'outflow': []}

        interval = 15

        img_num =len(gts.keys())
        for img_id in list(gts.keys()):
            print(img_id)
            if img_id>img_num-interval:
                break

            img_id_b = img_id+interval

            pre_ids,pre_ids_b = pred[img_id],pred[img_id_b]

            gt_ids,gt_ids_b = gts[img_id], gts[img_id_b]

            # print(gt_ids, gt_ids_b)

            if img_id == 1:
                pred_dict['first_frame'] = len(pre_ids)
                gt_dict['first_frame'] = len(gt_ids)

            def generate_cycle_mask(height, width):
                        x, y = np.ogrid[-height:height + 1, -width:width + 1]
                        # ellipse mask
                        mask = ((x) ** 2 / (height ** 2) + (y) ** 2 / (width ** 2) <= 1)
                        mask.dtype = 'uint8'
                        return mask
            def save_inflow_outflow_density(img, pre_points_a, pre_points_b, inflow_idx,outflow_idx, save_path, scene_id, vi, intervals):
                img_w,img_h = img.size
                pre_inflow = np.zeros((img_h, img_w))
                pre_outflow = np.zeros((img_h, img_w))

                # matched_mask = np.zeros(scores.shape)
                # matched_mask[match_gt['a2b'][:, 0], match_gt['a2b'][:, 1]] = 1
                # matched_mask[match_gt['un_a'], -1] = 1
                # matched_mask[-1, match_gt['un_b']] = 1

                kernel = 8
                wide = 2 * kernel + 1
                mask = generate_cycle_mask(kernel, kernel)
                for row_id in outflow_idx:
                    # import pdb
                    # pdb.set_trace()
                    pos = pre_points_a[row_id]
                    w, h = int(pos[0]),int(pos[1])
                    h_min, h_max = max(0, h - kernel), min(img_h, h + kernel + 1)
                    w_min, w_max = max(0, w - kernel), min(img_w, w + kernel + 1)
                    pre_outflow[h_min:h_max, w_min:w_max] = mask[max(kernel - h, 0):wide - max(0, kernel + 1 + h - img_h),
                                                            max(kernel - w, 0):wide - max(0, kernel + 1 + w - img_w)] * 1.
                pred_outflow_map = cv2.applyColorMap((255 * pre_outflow / (pre_outflow.max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)

                for column_id in inflow_idx:
                    pos = pre_points_b[column_id]
                    w, h = int(pos[0]),int(pos[1])
                    h_min, h_max = max(0, h - kernel), min(img_h, h + kernel + 1)
                    w_min, w_max = max(0, w - kernel), min(img_w, w + kernel + 1)
                    pre_inflow[h_min:h_max, w_min:w_max] = mask[max(kernel - h, 0):wide - max(0, kernel + 1 + h - img_h),
                                                            max(kernel - w, 0):wide - max(0, kernel + 1 + w - img_w)] * 1.

                pred_inflow_map = cv2.applyColorMap((255 * pre_inflow / (pre_inflow.max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)


                os.makedirs(save_path, mode=0o777, exist_ok=True)
                stem = '{}_{}_{}_matches_outflow_pre_{}'.format(scene_id, vi, vi + intervals, len(pre_outflow))
                out_file = os.path.join(save_path, stem + '.png')
                # print('\n Writing image to {}'.format(out_file))
                cv2.imwrite(out_file, pred_outflow_map)

                stem = '{}_{}_{}_matches_inflow_pre_{}'.format(scene_id, vi, vi + intervals,len(pre_inflow))
                out_file =  os.path.join(save_path, stem + '.png')
                # print('\n Writing image to {}'.format(out_file))
                cv2.imwrite(out_file, pred_inflow_map)

            #
            # import pdb
            # pdb.set_trace()
            # print(pre_ids, pre_ids_b)
            dis = torch.tensor(pre_ids).unsqueeze(1).expand(-1, len(pre_ids_b)) - torch.tensor(pre_ids_b).unsqueeze(0).expand(len(pre_ids), -1)
            dis = dis.abs()
            if dis.size(0) ==0 or dis.size(1)==0:
                continue
            matched_a, matched_b = torch.where(dis == 0)
            matched_a2b = torch.stack([matched_a, matched_b], 1)
            unmatched0 = torch.where(dis.min(1)[0] > 0)[0]
            unmatched1 = torch.where(dis.min(0)[0] > 0)[0]

            pre_inflow =unmatched1
            pre_outflow = unmatched0



            img_name = ('%03d' % img_id) + '.jpg'
            img_path = osp.join(dataset_root, 'video_ori',scene_name, img_name)
            if not os.path.exists(img_path):
                img_name = ('%04d' % img_id) + '.jpg'
                img_path = osp.join(dataset_root, 'video_ori', scene_name, img_name)
            if not os.path.exists(img_path):
                img_name = ('%05d' % img_id) + '.jpg'
                img_path = osp.join(dataset_root, 'video_ori', scene_name, img_name)
            if not os.path.exists(img_path):
                img_name = ('%06d' % img_id) + '.jpg'
                img_path = osp.join(dataset_root, 'video_ori', scene_name, img_name)
            if not os.path.exists(img_path):
                img_name = ('%07d' % img_id) + '.jpg'
                img_path = osp.join(dataset_root, 'video_ori', scene_name, img_name)
            if not os.path.exists(img_path):
                img_name = ('%08d' % img_id) + '.jpg'
                img_path = osp.join(dataset_root, 'video_ori', scene_name, img_name)
            print(img_path)
            img = Image.open(img_path)


            pre_points_a, pre_points_b = detect_points[img_id], detect_points[img_id_b]
            save_inflow_outflow_density(img, pre_points_a, pre_points_b, pre_inflow, pre_outflow,
                                        osp.join('/media/D/GJY/ht/CVPR2022/dataset/demo_tracking',scene_name), scene_name,img_id, intervals=interval)

            # print(pre_inflow,pre_outflow)
            gt_inflow = set(gt_ids_b)-set(gt_ids)
            gt_outflow = set(gt_ids)-set(gt_ids_b)
            pred_dict['inflow'].append(len(pre_inflow))
            pred_dict['outflow'].append(len(pre_outflow))
            print('pre', len(pre_inflow), len(pre_outflow))
            gt_dict['inflow'].append(len(gt_inflow))
            gt_dict['outflow'].append(len(gt_outflow))
            print('gt',len(gt_inflow), len(gt_outflow))
        # import pdb
        # pdb.set_trace()
        scenes_pred_dict['all'].append(pred_dict)
        scenes_gt_dict['all'].append(gt_dict)

        scene_l = scene_label[i.split('.')[0]]

        if scene_l[0] == 0: scenes_pred_dict['in'].append(pred_dict);  scenes_gt_dict['in'].append(gt_dict)
        if scene_l[0] == 1: scenes_pred_dict['out'].append(pred_dict);  scenes_gt_dict['out'].append(gt_dict)
        if scene_l[1] == 0: scenes_pred_dict['day'].append(pred_dict);  scenes_gt_dict['day'].append(gt_dict)
        if scene_l[1] == 1: scenes_pred_dict['night'].append(pred_dict);  scenes_gt_dict['night'].append(gt_dict)
        if scene_l[2] == 0: scenes_pred_dict['scenic0'].append(pred_dict);  scenes_gt_dict['scenic0'].append(gt_dict)
        if scene_l[2] == 1: scenes_pred_dict['scenic1'].append(pred_dict);  scenes_gt_dict['scenic1'].append(gt_dict)
        if scene_l[2] == 2: scenes_pred_dict['scenic2'].append(pred_dict);  scenes_gt_dict['scenic2'].append(gt_dict)
        if scene_l[2] == 3: scenes_pred_dict['scenic3'].append(pred_dict);  scenes_gt_dict['scenic3'].append(gt_dict)
        if scene_l[2] == 4: scenes_pred_dict['scenic4'].append(pred_dict);  scenes_gt_dict['scenic4'].append(gt_dict)
        if scene_l[2] == 5: scenes_pred_dict['scenic5'].append(pred_dict);  scenes_gt_dict['scenic5'].append(gt_dict)
        if scene_l[3] == 0: scenes_pred_dict['density0'].append(pred_dict);  scenes_gt_dict['density0'].append(gt_dict)
        if scene_l[3] == 1: scenes_pred_dict['density1'].append(pred_dict);  scenes_gt_dict['density1'].append(gt_dict)
        if scene_l[3] == 2: scenes_pred_dict['density2'].append(pred_dict);  scenes_gt_dict['density2'].append(gt_dict)
        if scene_l[3] == 3: scenes_pred_dict['density3'].append(pred_dict);  scenes_gt_dict['density3'].append(gt_dict)
        if scene_l[3] == 4: scenes_pred_dict['density4'].append(pred_dict);  scenes_gt_dict['density4'].append(gt_dict)
        # print(pred_dict, gt_dict)
        # scenes_pred_dict.append(pred_dict)
        # scenes_gt_dict.append(gt_dict)
    # import pdb
    # pdb.set_trace()
    for key in scenes_pred_dict.keys():
        s_pred_dict = scenes_pred_dict[key]
        s_gt_dict = scenes_gt_dict[key]
        MAE, MSE, WRAE, MIAE, MOAE, cnt_result = compute_metrics_all_scenes(s_pred_dict, s_gt_dict, interval)
        if key == 'all':save_cnt_result = cnt_result

        print('='*20, key, '='*20)
        print('MAE: %.2f, MSE: %.2f  WRAE: %.2f WIAE: %.2f WOAE: %.2f' % (MAE.data, MSE.data, WRAE.data, MIAE.data, MOAE.data))
    print(save_cnt_result)
if __name__ == '__main__':

    LOI_metrics_in_sense()
    # tracking_to_crowdflow()