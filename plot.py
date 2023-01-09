import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

x = [0,1,2]
sea_3_18 = [20,21.33,32.8]
sea_3_50 = [21.6,25.07,34.4]
sea_21_18 = [19.47,24.53,32]
sea_21_50 = [19.73,22.13,30.93]

mae_3_18 = [1.984,1.677,1.312]
mae_3_50 = [1.832,1.549,1.234]
mae_21_18 = [1.893,1.554,1.416]
mae_21_50 = [1.984,1.76,1.434]

plt.plot(x, sea_3_18, linewidth=1, label="3D-ResNet-18")
plt.plot(x, sea_3_50, linewidth=1, label="3D-ResNet-50")
plt.plot(x, sea_21_18, linewidth=1, label="(2+1)D-ResNet-18")
plt.plot(x, sea_21_50, linewidth=1, label="(2+1)D-ResNet-50")
x_ticks = ['@0','@50','TFS']

plt.xticks(x, x_ticks)
plt.ylabel('SEA%')
plt.title('SEA')
plt.legend()
plt.show()

plt.plot(x, mae_3_18, linewidth=1, label="3D-ResNet-18")
plt.plot(x, mae_3_50, linewidth=1, label="3D-ResNet-50")
plt.plot(x, mae_21_18, linewidth=1, label="(2+1)D-ResNet-18")
plt.plot(x, mae_21_50, linewidth=1, label="(2+1)D-ResNet-50")
x_ticks = ['@0','@50','TFS']

plt.xticks(x, x_ticks)
plt.ylabel('MAE')
plt.title('MAE')
plt.legend()
plt.show()

t_loss_3_18 = [1.8292112328710348, 1.817167474405609, 1.8024998616128072, 1.7886049116614962, 1.773677297752269, 1.761133148722405, 1.7557786594342142, 1.7466440801202816, 1.7471810161632344, 1.7434447999418217, 1.7222105055829904, 1.7288727351348765, 1.7268274978999674, 1.7193533195196278, 1.6957477352915018, 1.688740389190451, 1.6693253390980463, 1.651594653181786, 1.6213947913072404, 1.5904886135219658, 1.5528783319640334, 1.5248156892557214, 1.4612000636810805, 1.4148780331559425, 1.337308081397175, 1.254550569683966, 1.1592043036744542, 1.0659630248581407, 0.934334895145284, 0.8853218420578616, 0.7751812984908584, 0.7628942750883798, 0.6634627397913132, 0.6272010282225853, 0.6029226192592705, 0.6491165473295825, 0.6611298175841352, 0.6847026555316291, 0.5753678604960442, 0.53005538837318, 0.5558638013083569, 0.5902246692537392, 0.5425813572793982, 0.5543375169708781, 0.5689618151879658, 0.5697058784048052, 0.5205799652277118, 0.5360284274729499, 0.5664001628212685, 0.5200664544083776, 0.519766761710609, 0.538874963828682, 0.49159837525038824, 0.48175907929013245, 0.4664389786385272, 0.46740515160299567, 0.4820939241534602, 0.651365896863659, 0.5638668249449591, 0.5118925009113159, 0.486156701769707, 0.5318142794018245, 0.5599313261408876, 0.5110912694439401, 0.4605167632664207, 0.4565178529515754, 0.4483115798147925, 0.44779835331396467, 0.4469069353964207, 0.44228165351996457, 0.443350653905068, 0.44322601243527265, 0.4461587486367156, 0.44482799630313025, 0.8362409289192109, 0.6388884857840782, 0.49442704690851436, 0.48937827702203807, 0.47286346978949806, 0.4690551898122704, 0.5555788432485866, 0.529881851190198, 0.5024349750393499, 0.45265833856741877, 0.4420129994515085, 0.44077063569404784, 0.4458281129163547, 0.4461832244465821, 0.44138506250659915, 0.44102924170285246, 0.44422052371023346, 0.7540676886909199, 0.5236843749336952, 0.45106680794571435, 0.4419853055074702, 0.4422495917138392, 0.44009503994109855, 0.4399901185810131, 0.43658310033544134, 0.4373062421044294]
t_loss_3_50 = [1.870499354209343, 1.8644236126085267, 1.843067577285488, 1.8282751358338516, 1.8139856074848315, 1.808908918478193, 1.805498892373412, 1.797464074879667, 1.7912942014471458, 1.7783634523405647, 1.7800809993361035, 1.7764769805608875, 1.7665441436489133, 1.76596037227742, 1.7622463494321725, 1.7443844362767074, 1.7488171514803477, 1.7402227255549745, 1.734572478889549, 1.7090203548869947, 1.7234481107579531, 1.714597863002415, 1.7058586104072793, 1.694771680953729, 1.6823807611082593, 1.6731335512042915, 1.6568223390265973, 1.6307511146921312, 1.6425262476405957, 1.647541081818351, 1.6045733642404096, 1.5857822479557817, 1.581685541320021, 1.532884524468958, 1.5616768722551582, 1.5086556205349246, 1.454024300740583, 1.3835013569706547, 1.3845903112070403, 1.3249873536346604, 1.244822108615054, 1.2370976407597536, 1.1494367097633598, 1.0506162363899885, 0.9759728860681074, 0.9434025683542238, 0.8850194412100054, 0.7986738528430897, 0.7626344621399023, 0.6849933361267522, 0.6382348259543851, 0.6728758993592575, 0.6545088263976313, 0.6623544061597246, 0.6009869053416008, 0.5506838890757874, 0.5851752292500795, 0.6712899111265683, 0.6142525863255898, 0.5733213926645091, 0.5986499032073648, 0.5574903888967786, 0.5195168330721611, 0.527567894028051, 0.5450023369129448, 0.5557528455762097, 0.7275512551304197, 0.6045136583873826, 0.5704050392565065, 0.5095102846840002, 0.4743390608116658, 0.47927330717118116, 0.6123236746248537, 0.5638150857204068, 0.5047576656415514, 0.5023919585412436, 0.4936956178641667, 0.517499729773424, 0.6090393601542842, 0.6013334991723082, 0.518454327744289, 0.49349936451355036, 0.4840819192302488, 0.5044317641397462, 0.488838242353314, 0.4930708880950935, 0.492947013926332, 0.5898379666091752, 0.5556905691641091, 0.479328355378043, 0.4600367590132421, 0.4529577468651055, 0.453858828076916, 0.44754506348475925, 0.4487561411666174, 0.4467832222037072, 0.4450303952406793, 0.44902510614725794, 0.581289952036238, 0.8891757236127436]
t_loss_21_18 = [1.824079829410915, 1.786946296256824, 1.7702680830537838, 1.7602844177371395, 1.7427809786622541, 1.7214729215976965, 1.7305039193508398, 1.7039397137008445, 1.7054309422952416, 1.7038575802406255, 1.6879940228740664, 1.699445295942961, 1.6729789480675745, 1.6666009663230312, 1.6587613877588816, 1.621238451804558, 1.6075558244746968, 1.5940562854283047, 1.5547236541326899, 1.519206170618099, 1.4625808966420863, 1.4287089679363, 1.3363414972367949, 1.335256604817662, 1.2825778081034223, 1.2176430598445183, 1.1804003824282736, 1.1199058223597325, 1.0516839503806874, 0.9696458833713602, 0.9131459937478504, 0.8752915458413806, 0.8643645873687563, 0.7744007327043227, 0.7941067459156913, 0.7526553334110845, 0.6662101979455809, 0.6895709430431798, 0.6752717430791716, 0.6407048855058468, 0.6003838439818716, 0.6540657337347087, 0.598524287941247, 0.5889560737966621, 0.6087137503127982, 0.5838251813811107, 0.5827144062975897, 0.6764780484738141, 0.5936334414856277, 0.5856440612216935, 0.5500993359371693, 0.5353021912113594, 0.552635999491615, 0.5515306027483766, 0.6019071498817771, 0.5460332014264851, 0.5369540187564209, 0.5704927611960112, 0.539643156430582, 0.5107628738466841, 0.48661067760991356, 0.5516456186662625, 0.6026239637693349, 0.5370556061176488, 0.5081258142081491, 0.4929974292860414, 0.5757101062875595, 0.5719535313274738, 0.49344488439986306, 0.5316407751754253, 0.5239823350723642, 0.5183236199247576, 0.4696633593879477, 0.5047788110418912, 0.5696329643038938, 0.5107146418877762, 0.5407351232031836, 0.5094072735005052, 0.5087282280199719, 0.49860504394682653, 0.5024077147789245, 0.50852952429848, 0.4731393282430886, 0.4532366023011451, 0.4477796515409094, 0.4468647547229363, 0.4470441145618467, 0.4455951153144349, 0.4524578939932976, 0.759359807265501, 0.5958958653202893, 0.568813151414812, 0.484851174328449, 0.46189762461577016, 0.45413704559098195, 0.4492821834274452, 0.4406972993029295, 0.44168467390058685, 0.4435861118300988, 0.44401512441843966]
t_loss_21_50 = [1.8497679333617216, 1.8178898787846531, 1.794016755845425, 1.779119457206587, 1.7667238703609383, 1.7640605621094252, 1.7457314233710295, 1.7407449836278484, 1.7309302448356239, 1.7332755931972588, 1.7205394275867156, 1.7249256646546134, 1.7109957533161135, 1.6908475521707187, 1.6795013333759168, 1.6617875647370832, 1.6846032980149679, 1.6436772385652918, 1.6384504937777553, 1.6295371512426948, 1.638116526777727, 1.6074007762609608, 1.5788317529389457, 1.565119373537328, 1.5371244486666074, 1.51523030652617, 1.497189088894503, 1.4611177796865031, 1.432848255999767, 1.3874913251748051, 1.3536718207119156, 1.309187027224659, 1.2570607999380488, 1.2368222678447292, 1.1819556706795726, 1.1198822461123015, 1.0678629507548618, 1.015303199308632, 0.9237158767280788, 0.8332870833195039, 0.8574296549288896, 0.8215955399031186, 0.7761334949818841, 0.7896891463085682, 0.7273603709292238, 0.7518722157191186, 0.6650165769740612, 0.6840106510985506, 0.6906147291407968, 0.6910007429601502, 0.618487766134913, 0.6297638460341162, 0.6217056746139143, 0.6435059713077371, 0.6191997861448866, 0.5787656253380496, 0.6064201349759624, 0.6960688635271831, 0.5877711888756195, 0.5763145422522169, 0.5616446990166267, 0.6235672796729708, 0.6671268935077381, 0.5767347921420188, 0.5388819559538451, 0.5439567030509458, 0.567017335145578, 0.5732241909488709, 0.544652843507972, 0.5500265982573049, 0.6319922551947789, 0.5533975673853045, 0.5584928597847041, 0.5645420554019239, 0.5297873761967151, 0.5932456053739047, 0.6043585307189148, 0.5785348390902045, 0.5424713188388052, 0.4997342500264627, 0.4962388193846619, 0.47441334375282274, 0.47180015436054146, 0.5476471628912174, 0.6160254927664778, 0.5941638696911561, 0.510492633865045, 0.48289499323080926, 0.45892244063480925, 0.45719873470111483, 0.5109524741007464, 0.677400146033207, 0.5583029266474021, 0.5449932003325789, 0.5187675589083756, 0.48949120076794694, 0.537385444843421, 0.4928104312424242, 0.5111855884103009, 0.5167075518491494]
base_loss = [1.7973541034399159, 1.7934701634149481, 1.7911584229364883, 1.7884700089475534, 1.7846482935613088, 1.7787256249546135, 1.7775754793717042, 1.7732462983061796, 1.7690424579773507, 1.772638364864962, 1.7725034134231346, 1.7758113837590184, 1.7659648487167636, 1.7623930286316976, 1.7562790974213258, 1.7554894516067783, 1.7457939577798773, 1.753290420466096, 1.7421351814792103, 1.7400756700195534, 1.7310489316056246, 1.7369978158143315, 1.7300354834020573, 1.7233938344203643, 1.7144357301022886, 1.7118076020783752, 1.7188160824079584, 1.7015690459822217, 1.7030988305154509, 1.6945695807463932, 1.686183540490422, 1.6845370705110314, 1.6841054515246927, 1.6841142090567707, 1.6627922084209692, 1.6569838236718282, 1.646332408389906, 1.6311051040670297, 1.6197251829787762, 1.6011721130705228, 1.5959609311862584, 1.5809118121209806, 1.5725249032469562, 1.5593015181757237, 1.5505680006785985, 1.5440402829299007, 1.5244092221242669, 1.5126382225621355, 1.506948043612668, 1.4947045719536551, 1.486324106472252, 1.4864199714068949, 1.4629124168061862, 1.466848640111241, 1.4500684786016924, 1.4456366822667366, 1.4204642681309776, 1.4271443684170717, 1.4032271051928944, 1.391631077893459, 1.4006343000996722, 1.3943574972396349, 1.387517372404572, 1.3747908235901463, 1.3666685223579407, 1.3444224702615808, 1.3513314700039634, 1.334748165450827, 1.3238873729740617, 1.322330460061122, 1.316755745750274, 1.2860944543006647, 1.2816215053091955, 1.2881805268082305, 1.2555350424164402, 1.2720856986341684, 1.2323678109767664, 1.2474959890772825, 1.2204417316582952, 1.2161046514545915, 1.1932713911481148, 1.1781741902776008, 1.1805240849073786, 1.1760758505250415, 1.166273412260696, 1.1536665644741406, 1.154688333507872, 1.1529528642222828, 1.132944664163311, 1.1321512023462867, 1.1023609359769055, 1.0897098389202662, 1.1031574156597583, 1.0727371866033024, 1.0460559573486774, 1.0663579126996716, 1.053061312458811, 1.0315549633581274, 1.0359190973269679, 0.9796473755888695]


t_acc_3_18 = [0.22071602363573198, 0.22549530761209627, 0.22749391727493964, 0.2436565867222805, 0.27181091414668107, 0.2771115745568302, 0.27659019812304525, 0.28475842891901343, 0.2963156065345846, 0.29457768508863436, 0.3006604101494617, 0.3013555787278418, 0.29527285366701467, 0.2930135557872787, 0.3253388946819607, 0.32551268682655593, 0.3463677441779637, 0.3567952728536675, 0.3704379562043799, 0.4127563434132782, 0.4047619047619049, 0.4381299965241574, 0.4969586374695866, 0.5101668404588117, 0.5582203684393467, 0.5868960722975324, 0.641988182134169, 0.6902155022592997, 0.7638164754953094, 0.8012686826555463, 0.8561001042752883, 0.8534932221063627, 0.9051094890510968, 0.9175356273896434, 0.9247480013903385, 0.9013729579423024, 0.8966805700382361, 0.8794751477233246, 0.9254431699687188, 0.9405630865484891, 0.9279631560653472, 0.9176225234619412, 0.9295272853667024, 0.9248348974626363, 0.9170142509558583, 0.9184914841849162, 0.9400417101147036, 0.9347410497045545, 0.9179701077511311, 0.9347410497045547, 0.935783802572125, 0.929440389294405, 0.9473409801876965, 0.9489051094890517, 0.9509906152241928, 0.9556830031282596, 0.9461244351755309, 0.8879040667361854, 0.9238790406673631, 0.9389120611748357, 0.9473409801876963, 0.9259645464025039, 0.9185783802572139, 0.9405630865484891, 0.9520333680917632, 0.955161626694474, 0.9567257559958295, 0.9535974973931184, 0.9535974973931182, 0.9566388599235319, 0.9541188738269037, 0.952033368091763, 0.9582029892248877, 0.9603753910323258, 0.8154327424400442, 0.885297184567259, 0.9430830726451176, 0.9478623566214818, 0.947254084115399, 0.9483837330552667, 0.9212721584984371, 0.9316127911018436, 0.9436913451511999, 0.9498609662843248, 0.9530761209593334, 0.951511991657978, 0.9529892248870356, 0.9530761209593336, 0.9577685088633996, 0.95142509558568, 0.9493395898505392, 0.848800834202295, 0.931091414668058, 0.955161626694474, 0.9566388599235318, 0.9530761209593331, 0.9509037191518949, 0.9545533541883913, 0.9577685088634, 0.9535974973931186]
t_acc_3_50 = [0.21159193604449095, 0.2059436913451514, 0.2164581160931529, 0.22332290580465786, 0.2320994091067086, 0.23844282238442852, 0.24608967674661142, 0.2458289885297188, 0.25451859575947217, 0.25773375043448044, 0.2629475147723325, 0.2639033715676056, 0.27997914494264886, 0.27328814737573925, 0.2748522766770947, 0.2957942301008, 0.2936218282933615, 0.2917970107751135, 0.28632255822036873, 0.32464372610358067, 0.29466458116093186, 0.32125477928397667, 0.30848105665623976, 0.31091414668057044, 0.31578032672923234, 0.3469760166840463, 0.35418839068474145, 0.36409454292666044, 0.3488877302745925, 0.35592631213069226, 0.37043795620438, 0.3817344456030592, 0.38069169273548864, 0.4042405283281201, 0.41718804310045204, 0.4260514424748006, 0.45107751129649004, 0.5036496350364962, 0.50808133472367, 0.5372784150156418, 0.5781195689954812, 0.581160931525895, 0.6427702467848462, 0.710549183176921, 0.7469586374695878, 0.7615571776155732, 0.7950990615224214, 0.8404588112617333, 0.8544490789016355, 0.8894681960375408, 0.9092805005213783, 0.8915537017726818, 0.894681960375393, 0.9009384775808149, 0.9196211331247841, 0.9353493222106373, 0.9223149113660076, 0.8794751477233247, 0.9087591240875927, 0.9285714285714298, 0.9091936044490805, 0.9290928050052149, 0.9374348279457778, 0.9369134515119927, 0.9301355578727855, 0.9249217935349335, 0.8540145985401477, 0.9065867222801547, 0.9202294056308666, 0.9435175530066053, 0.9519464720194653, 0.9467327076816134, 0.8920750782064671, 0.9228362877997928, 0.9421272158498443, 0.9431699687174149, 0.9410844629822743, 0.9306569343065706, 0.9014598540146, 0.9082377476538073, 0.9356969064998272, 0.9420403197775472, 0.9436913451512001, 0.9389989572471334, 0.9416058394160595, 0.9436913451512001, 0.9435175530066051, 0.9071949947862372, 0.921619742787627, 0.9452554744525555, 0.9499478623566225, 0.9529892248870359, 0.9546402502606888, 0.954640250260689, 0.9551616266944739, 0.9515119916579777, 0.9519464720194653, 0.9541188738269037, 0.8997219325686493, 0.782933611400766]
t_acc_21_18 = [0.20976711852624289, 0.2691171359054574, 0.27598192561696233, 0.2911018421967332, 0.2889294403892948, 0.30656934306569406, 0.295359749739312, 0.3136079249217939, 0.3322905804657631, 0.30787278415015656, 0.3202989224887037, 0.31664928745220755, 0.33289885297184624, 0.34367396593674027, 0.34541188738269063, 0.3749565519638517, 0.3755648244699348, 0.38129996524157184, 0.41936044490789026, 0.4183176920403204, 0.46367744177963194, 0.48618352450469254, 0.5325860271115749, 0.511904761904762, 0.5565693430656936, 0.5962808481056662, 0.6293882516510263, 0.6572818908585343, 0.6911713590545716, 0.7450469238790421, 0.7693778241223516, 0.7798053527980551, 0.7949252693778265, 0.8424574209245759, 0.8345498783455009, 0.8528849496002797, 0.9001564129301367, 0.8762599930483164, 0.8873826903024, 0.9050225929787988, 0.9202294056308669, 0.8962460896767481, 0.9232707681612806, 0.925877650330206, 0.906673618352452, 0.9176225234619408, 0.9212721584984371, 0.8706986444212743, 0.9159714980882879, 0.9171011470281556, 0.9336982968369841, 0.9336982968369841, 0.9310914146680579, 0.9353493222106373, 0.9113660062565185, 0.9305700382342732, 0.9301355578727852, 0.9191866527632968, 0.932568647897116, 0.9447340980187705, 0.9525547445255482, 0.9295272853667027, 0.8978102189781034, 0.931091414668058, 0.9430830726451174, 0.9452554744525558, 0.9138859923531469, 0.9077163712200221, 0.9473409801876963, 0.9343065693430669, 0.9347410497045543, 0.935870698644422, 0.9530761209593336, 0.940041710114704, 0.9207507820646517, 0.9357838025721249, 0.9290928050052147, 0.9379562043795631, 0.934827945776852, 0.940041710114704, 0.9400417101147044, 0.9405630865484891, 0.951511991657978, 0.9556830031282594, 0.9571602363573171, 0.9593326381647554, 0.9608967674661113, 0.9545533541883913, 0.9598540145985404, 0.8408063955509228, 0.9035453597497406, 0.912930135557874, 0.9489051094890519, 0.955161626694474, 0.9525547445255482, 0.9556830031282594, 0.957247132429615, 0.9582898852971853, 0.9577685088634003, 0.9546402502606888]
t_acc_21_50 = [0.2092457420924576, 0.22844977407021225, 0.24939172749391758, 0.27389641988182173, 0.2651199165797711, 0.2748522766770946, 0.28701772679874926, 0.28779979144942686, 0.3077858880778594, 0.28623566214807117, 0.3089155370177272, 0.3067431352102888, 0.3071776155717763, 0.32916232186305233, 0.3223844282238447, 0.34297879735835973, 0.31404240528328176, 0.35245046923879114, 0.35462287104622936, 0.3453249913103933, 0.34836635384080683, 0.37052485227667764, 0.388077858880779, 0.3979840111226974, 0.4111922141119225, 0.4278762599930488, 0.4407368786930836, 0.46515467500869007, 0.4933958985053873, 0.4966979492526938, 0.5296315606534585, 0.5591762252346199, 0.5679527285366697, 0.5926312130691692, 0.6279979144942651, 0.6650156412930139, 0.6857838025721251, 0.711505039972194, 0.767205422314913, 0.8089155370177291, 0.7974452554744543, 0.8189954814042426, 0.838199513381997, 0.8314216197427895, 0.8564476885644794, 0.8501911713590564, 0.886253041362532, 0.8809523809523828, 0.8717413972888446, 0.8800834202294072, 0.9144942648592299, 0.9071949947862372, 0.9071949947862371, 0.8915537017726813, 0.9035453597497412, 0.9159714980882875, 0.9040667361835263, 0.8659193604449099, 0.9207507820646519, 0.9180570038234283, 0.9228362877997927, 0.9003302050747323, 0.8811261730969775, 0.9223149113660072, 0.9352624261383395, 0.929614181439, 0.9232707681612806, 0.9091936044490804, 0.9310914146680583, 0.9254431699687186, 0.8883385470976726, 0.9233576642335773, 0.9258776503302063, 0.9223149113660073, 0.9374348279457778, 0.9075425790754273, 0.9019812304483854, 0.9149287452207175, 0.9275286757038596, 0.9383906847410507, 0.9441258255126878, 0.9513381995133827, 0.9472540841153986, 0.9237921445950652, 0.8987660757733765, 0.9071080987139394, 0.9353493222106373, 0.9452554744525555, 0.9509906152241925, 0.9525547445255481, 0.9378693083072656, 0.87434827945777, 0.9176225234619411, 0.9207507820646517, 0.9348279457768518, 0.9447340980187703, 0.9285714285714298, 0.9447340980187707, 0.9383906847410507, 0.9337851929092817]
base_acc = [0.21115745568300356, 0.22949252693778283, 0.20959332638164796, 0.24148418491484228, 0.2337504344803618, 0.2415710809871397, 0.23983315954118903, 0.2442648592283632, 0.2659019812304486, 0.2509558567952734, 0.25139033715676085, 0.2556482446993399, 0.2628606187000351, 0.2453076120959337, 0.27068126520681307, 0.2760688216892601, 0.2785888077858883, 0.2759819256169625, 0.2864963503649641, 0.2822384428223846, 0.2852798053527984, 0.27537365311088, 0.2916232186305181, 0.2966631908237753, 0.30039972193256886, 0.300399721932569, 0.2915363225582207, 0.3030935001737926, 0.3019638512339245, 0.30830726451164475, 0.31969064998262114, 0.3076989920055617, 0.3222975321515471, 0.3183003128258605, 0.3274244004171014, 0.33811261730969794, 0.3328119568995488, 0.34532499131039307, 0.3410670837678141, 0.3548835592631221, 0.35531803962460945, 0.35114702815432786, 0.38199513381995154, 0.363573166492875, 0.3739137990962813, 0.39807090719499527, 0.4053701772679878, 0.39963503649635096, 0.39485575251998645, 0.396941258255127, 0.4133646159193609, 0.41797010775112975, 0.4386513729579426, 0.41449426485922886, 0.4439520333680921, 0.4313521028849498, 0.4569864442127219, 0.45959332638164757, 0.46784845324991337, 0.4741049704553356, 0.46002780674313537, 0.4752346193952037, 0.47280152937087283, 0.4882690302398333, 0.4934827945776849, 0.5042579075425787, 0.5039103232533888, 0.5261557177615573, 0.5008689607229755, 0.5184219673270769, 0.5220716023635732, 0.5351929092805005, 0.5325860271115741, 0.5326729231838722, 0.5557003823427185, 0.5507473062217593, 0.5737747653806053, 0.5508342022940562, 0.5901112269725411, 0.5903719151894339, 0.5759471671880434, 0.6020159888773036, 0.5941084462982271, 0.6081856100104278, 0.6020159888773031, 0.6087938825165106, 0.6259993048314231, 0.6279979144942656, 0.6316475495307613, 0.6291275634341329, 0.6510253736531112, 0.6662321863051801, 0.6552832811956908, 0.6817865832464381, 0.6980361487660768, 0.6713590545707343, 0.6969933958985074, 0.6958637469586383, 0.7009906152241927, 0.7272332290580479]

def smooth(y, window, poly=1):
    return savgol_filter(y,window,poly)

plt.plot(smooth(t_loss_3_18,window=35), linewidth=1, label="3D-ResNet-18")
plt.plot(smooth(t_loss_3_50,window=35), linewidth=1, label="3D-ResNet-50")
plt.plot(smooth(t_loss_21_18,window=35), linewidth=1, label="(2+1)D-ResNet-18")
plt.plot(smooth(t_loss_21_50,window=35), linewidth=1, label="(2+1)D-ResNet-50")
plt.plot(smooth(base_loss,window=35), linewidth=1, label="Plain CNN")

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training loss')
plt.legend()
plt.show()

plt.plot(smooth(t_acc_3_18,window=35), linewidth=1, label="3D-ResNet-18")
plt.plot(smooth(t_acc_3_50,window=35), linewidth=1, label="3D-ResNet-50")
plt.plot(smooth(t_acc_21_18,window=35), linewidth=1, label="(2+1)D-ResNet-18")
plt.plot(smooth(t_acc_21_50,window=35), linewidth=1, label="(2+1)D-ResNet-50")
plt.plot(smooth(base_acc,window=35), linewidth=1, label="Plain CNN")

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training accuracy')
plt.legend()
plt.show()