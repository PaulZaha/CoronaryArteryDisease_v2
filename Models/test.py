import numpy as np

list = [nan, nan, nan, 0.6828007923520681, 0.3014582164890591, 0.08562300319488454, 0.0831559528877513, 0.34806537691793954, 0.3291660072796224, 0.33228929384965517, 0.34057855202173015, nan, 0.27990482370754316, 0.545106179147356, 0.23177777777777261, nan, nan, 0.684397503600563, 0.35595105672967986, 0.6316808098853561, nan, nan, nan, 0.6756827731092259, 0.4441390143333813, 0.46143682906687167, 0.6335841203777712, 0.04268675455115954, 0.41133424098025173, 0.3381230354737163, 0.2804081632653023, 0.18962165395543024, 0.18140161725066897, 0.49579387979029954, 0.24400446179586605, 0.6237502192597736, nan, 0.49289428076255365, 0.5204043429427083, 0.11606334841628697, nan, 0.06177024482108996, 0.005331660655480466, nan, 0.1548962404359536, 0.45422327229768295, 0.30664972469291374, 0.10712263616804196, 0.1669317579338004, 0.17423664122137072, nan, nan, 0.4466807909604362, 0.2533013205282012, 0.24594931449937002, 0.15199547127087032, 0.3090582959641228, 0.34233697732782903, nan, 
0.3723508314313419, 0.10153294843718488, 0.24295355299721466, 0.07345240657054626, 0.3495920847075039, 0.16349031343518147, 0.005505308690522572, 0.0016501650165015595, 0.34578441835644197, 0.038872072646167315, 0.27253727184059434, 0.05271966527196211, 0.24937838233142826, 0.06309611151870688, 0.1987378329233052, 0.11801763964720209, nan, 0.21259842519684366, nan, nan, 0.20757229832571902, 0.17323709050526842, 0.08275479033601547, 0.2715950180795427, nan, 0.16993364350545084, 0.13616058133301406, 0.36974443528441375, 0.1813823411905568, 0.5674748398901974, 0.13002538278882944, 0.0662552776875566, 0.43446682073290077, nan, 0.4957614135430396, 0.3646127755794131, 0.5560561887092401, nan, 0.36102098204628247, 0.21706007243661252, 0.09836065573769848, 0.5012722646310274, 0.3834571097635705, 0.19357382219593167, 0.2751579970831218, 
nan, 0.6329843726293238, 0.30373161105130414, 0.03450862715678791, nan, 0.22197954929017727, 0.26218818712563097, 0.2531083481349836, nan, 0.08632917583761099, nan, 0.15531188207801408, 0.20305289314873498, 0.46389891696749225, 0.2590269537209609, 0.4435633095426792, 0.02055335968379365, nan, 0.6597353497164254, 0.02360931435963701, 0.4679300291545019, 0.5481184454040546, 0.5054355919582872, 0.15784123463635233, 0.41177890417570806, 0.4496487119437858, 0.23439849624059708, nan, nan, 0.294187830236915, 0.20744728577837224, 0.1786455530786351, 0.2808018729879971, 0.28715706964123777, 0.11283109735121964, 0.1549907005579646, 0.4045559647169581, 0.12649340354453173, 0.3775782098095257, 0.2188005637037401, nan, nan, 0.27661510464057604, 0.24071702944942056, 0.3156686435813981, 0.4190728476821118, 0.033661006851354376, 0.3889008388900671, nan, 0.06493506493506325, 0.20847997318585204, 0.23258282162079857, 0.3111374407582864, 0.20694752402069197, nan, 0.008300190212692229, nan, 0.3715585348335948, 0.005778069599474569, 0.022059442620189915, nan, nan, 0.294694153430193, 0.42132709944253877, 0.09036237471086844, nan, 0.6201208401265825, 0.4358830146231598, 0.6110025706940748, 0.14044592623462634, 0.0939145083175366, nan, nan, 0.5104834012812897, 0.34340693228073654, 0.49447825107052085, nan, 0.15502518064374185, 0.09352649917476447, 0.6437313026923949, 0.06368488864347271, 0.22249240121579192, nan, 0.26768060836500546, 0.13334537898635587, nan, 0.13788324961024984, 0.32324554228691893, 0.07386006783067268, 0.1894189891355602, 0.12821527502967608, nan, 0.34600521254605293, 0.20462397023650256, 0.3674456083803236, 0.14142678347934723, 0.22231292517006193, 0.3099900937285634, 0.3285845588235143, nan, nan, 0.6211883055642748, nan, 0.0003469210754553219, nan, 0.32605352199321286, 0.022940074906365966, 0.37965457138328695, 0.27013948192427356, 0.3685367485424759, 0.4128626871188244, 0.48208469055373765, nan, 0.2517215806759137, 0.06534870950027338, 0.10696721311474972, nan, 0.05273592386994239, nan, 0.12623464225487227, nan, 0.21154427319430577, 0.29166666666665764, 0.3893052461008657, nan, 0.4751822283929026, 0.32652354570636555, nan, 0.054775524354305605, nan, nan, 0.12286465177397353, 0.6432655232016847, 0.5454066165809794, nan, 0.0025067144136077884, nan, 0.018338427340927527, 0.15168983836328356, 0.31329810656097945, 0.04275996112730641, 0.15037593984961808, 0.04016064257027652, nan, 0.32129754778913605, 0.4023039432875141, 0.4861176470588006, 0.04713718457794166, 0.24181101011832473, 
0.30643158678915783, 0.4830132939438522, 0.5451361867704174, 0.004479910401791839, 0.5068539707714851, 0.5768253968253785, nan, 0.2764265129682933, 0.05239647470723042, 0.5084496693607492, nan, nan, 0.3982552626588205, 0.313959522573941, 0.03674614305750247, 0.5736806947227636, nan, 0.12467800103038386, 0.24609785119338914, nan, nan, nan, nan, 0.003415300546447854, 0.001068185864340357, 0.07820032137117533, nan, nan, 0.33110882956877996, nan, 0.37441282122132147, 0.04163675520459142, nan, 0.20033322225924424, 0.5580592773040893, 0.23168739791430504, 0.19739952718673787, 0.5273536460116326, 0.33126856173100844, 0.24540152534768586, nan, 0.09004204696953974, nan, 0.5233574584077353, nan, 0.44244688302590607, 0.5562422744128448]

arr_f1 = np.array(list)
cleaned_arr_f1 = arr_f1[~np.isnan(arr_f1)]