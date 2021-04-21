from cctpy import *

if __name__ == "__main__":
    DL2 = 2.1162209
    GAP3 = 0.1978111
    QS3_LEN = 0.2382791
    QS3_GRADIENT = -7.3733
    QS3_SECOND_GRADIENT = -45.31 * 2.0
    QS3_APERTURE = 60 * MM
    big_r_part2 = 0.95
    CCT_APERTURE = 80 * MM

    small_r_gap = 15 * MM
    small_r_innerest = 83 * MM
    agcct_small_r_in = small_r_innerest
    agcct_small_r_out = small_r_innerest + small_r_gap
    dipole_cct_small_r_in = small_r_innerest + small_r_gap * 2
    dipole_cct_small_r_out = small_r_innerest + small_r_gap * 3

    dipole_cct_winding_num: int = 128
    agcct_winding_nums: List[int] = [21, 50, 50]

    dipole_cct_bending_angle = 67.5
    dipole_cct_bending_rad = BaseUtils.angle_to_radian(dipole_cct_bending_angle)
    agcct_bending_angles: List[float] = [8 + 3.716404, 8 + 19.93897, 8 + 19.844626]
    agcct_bending_angles_rad: List[float] = BaseUtils.angle_to_radian(
        agcct_bending_angles
    )

    dipole_cct_tilt_angles = numpy.array([30.0, 80.0, 90.0, 90.0])
    agcct_tilt_angles = numpy.array([90.0, 30.0, 90.0, 90.0])

    dipole_cct_current = 9664.0
    agcct_current = -6000.0

    disperse_number_per_winding: int = 120

    trajectory_part2 = (
        Trajectory.set_start_point(P2(3.703795764767297, 1.5341624380266456))
        .first_line(P2(1, 1), DL2)
        .add_arc_line(0.95, True, dipole_cct_bending_angle)
        .add_strait_line(GAP3*2+QS3_LEN)
    )

    beamline = Beamline()
    # beamline.add(
    #     CCT.create_cct_along(
    #         trajectory=trajectory_part2,
    #         s=DL2,
    #         big_r=big_r_part2,
    #         small_r=dipole_cct_small_r_in,
    #         bending_angle=dipole_cct_bending_angle,
    #         tilt_angles=dipole_cct_tilt_angles,
    #         winding_number=dipole_cct_winding_num,
    #         current=dipole_cct_current,
    #         starting_point_in_ksi_phi_coordinate=P2.origin(),
    #         end_point_in_ksi_phi_coordinate=P2(
    #             2 * math.pi * dipole_cct_winding_num, -dipole_cct_bending_rad
    #         ),
    #         disperse_number_per_winding=disperse_number_per_winding,
    #     )
    # )

    # beamline.add(
    #     CCT.create_cct_along(
    #         trajectory=trajectory_part2,
    #         s=DL2,
    #         big_r=big_r_part2,
    #         small_r=dipole_cct_small_r_out,  # diff
    #         bending_angle=dipole_cct_bending_angle,
    #         tilt_angles=-dipole_cct_tilt_angles,  # diff ⭐
    #         winding_number=dipole_cct_winding_num,
    #         current=dipole_cct_current,
    #         starting_point_in_ksi_phi_coordinate=P2.origin(),
    #         end_point_in_ksi_phi_coordinate=P2(
    #             -2 * math.pi * dipole_cct_winding_num, -dipole_cct_bending_rad
    #         ),
    #         disperse_number_per_winding=disperse_number_per_winding,
    #     )
    # )

    agcct_index = 0
    agcct_start_in = P2.origin()
    agcct_start_out = P2.origin()
    agcct_end_in = P2(
        ((-1.0) ** agcct_index) * 2 * math.pi * agcct_winding_nums[agcct_index],
        -agcct_bending_angles_rad[agcct_index],
    )
    agcct_end_out = P2(
        ((-1.0) ** (agcct_index + 1)) * 2 * math.pi * agcct_winding_nums[agcct_index],
        -agcct_bending_angles_rad[agcct_index],
    )
    beamline.add(
        CCT.create_cct_along(
            trajectory=trajectory_part2,
            s=DL2,
            big_r=big_r_part2,
            small_r=agcct_small_r_in,
            bending_angle=agcct_bending_angles[agcct_index],
            tilt_angles=-agcct_tilt_angles,
            winding_number=agcct_winding_nums[agcct_index],
            current=agcct_current,
            starting_point_in_ksi_phi_coordinate=agcct_start_in,
            end_point_in_ksi_phi_coordinate=agcct_end_in,
            disperse_number_per_winding=disperse_number_per_winding,
        )
    )

    beamline.add(
        CCT.create_cct_along(
            trajectory=trajectory_part2,
            s=DL2,
            big_r=big_r_part2,
            small_r=agcct_small_r_out,
            bending_angle=agcct_bending_angles[agcct_index],
            tilt_angles=agcct_tilt_angles,
            winding_number=agcct_winding_nums[agcct_index],
            current=agcct_current,
            starting_point_in_ksi_phi_coordinate=agcct_start_out,
            end_point_in_ksi_phi_coordinate=agcct_end_out,
            disperse_number_per_winding=disperse_number_per_winding,
        )
    )

    for ignore in range(len(agcct_bending_angles) - 1):
        agcct_index += 1
        agcct_start_in = agcct_end_in + P2(
            0,
            -agcct_bending_angles_rad[agcct_index - 1]
            / agcct_winding_nums[agcct_index - 1],
        )
        agcct_start_out = agcct_end_out + P2(
            0,
            -agcct_bending_angles_rad[agcct_index - 1]
            / agcct_winding_nums[agcct_index - 1],
        )
        agcct_end_in = agcct_start_in + P2(
            ((-1) ** agcct_index) * 2 * math.pi * agcct_winding_nums[agcct_index],
            -agcct_bending_angles_rad[agcct_index],
        )
        agcct_end_out = agcct_start_out + P2(
            ((-1) ** (agcct_index + 1)) * 2 * math.pi * agcct_winding_nums[agcct_index],
            -agcct_bending_angles_rad[agcct_index],
        )
        beamline.add(
            CCT.create_cct_along(
                trajectory=trajectory_part2,
                s=DL2,
                big_r=big_r_part2,
                small_r=agcct_small_r_in,
                bending_angle=agcct_bending_angles[agcct_index],
                tilt_angles=-agcct_tilt_angles,
                winding_number=agcct_winding_nums[agcct_index],
                current=agcct_current,
                starting_point_in_ksi_phi_coordinate=agcct_start_in,
                end_point_in_ksi_phi_coordinate=agcct_end_in,
                disperse_number_per_winding=disperse_number_per_winding,
            )
        )

        beamline.add(
            CCT.create_cct_along(
                trajectory=trajectory_part2,
                s=DL2,
                big_r=big_r_part2,
                small_r=agcct_small_r_out,
                bending_angle=agcct_bending_angles[agcct_index],
                tilt_angles=agcct_tilt_angles,
                winding_number=agcct_winding_nums[agcct_index],
                current=agcct_current,
                starting_point_in_ksi_phi_coordinate=agcct_start_out,
                end_point_in_ksi_phi_coordinate=agcct_end_out,
                disperse_number_per_winding=disperse_number_per_winding,
            )
        )

    # Plot3.plot_line2(trajectory_part2)
    # Plot3.plot_beamline(beamline, ['r','y-']*3)
    # Plot3.set_center(trajectory_part2.point_at(DL2).to_p3(), 5)
    # Plot3.show()


    Plot2.plot_p2s(beamline.graident_field_along(trajectory_part2,step=10*MM),describe='r-')

    x = [0.0, 0.010050167747639907, 0.020100335495279814, 0.03015050324291972, 0.04020067099055963, 0.050250838738199535, 0.06030100648583944, 0.07035117423347935, 0.08040134198111926, 0.09045150972875916, 0.10050167747639907, 0.11055184522403898, 0.12060201297167888, 0.13065218071931878, 0.1407023484669587, 0.15075251621459862, 0.1608026839622385, 0.1708528517098784, 0.18090301945751833, 0.19095318720515825, 0.20100335495279814, 0.21105352270043803, 0.22110369044807796, 0.23115385819571788, 0.24120402594335777, 0.25125419369099766, 0.26130436143863756, 0.2713545291862775, 0.2814046969339174, 0.2914548646815573, 0.30150503242919724, 0.31155520017683713, 0.321605367924477, 0.3316555356721169, 0.3417057034197568, 0.35175587116739676, 0.36180603891503665, 0.37185620666267655, 0.3819063744103165, 0.3919565421579564, 0.4020067099055963, 0.4120568776532362, 0.42210704540087607, 0.432157213148516, 0.4422073808961559, 0.4522575486437958, 0.46230771639143575, 0.47235788413907565, 0.48240805188671554, 0.49245821963435543, 0.5025083873819953, 0.5125585551296352, 0.5226087228772751, 0.5326588906249151, 0.542709058372555, 0.5527592261201949, 0.5628093938678348, 0.5728595616154747, 0.5829097293631146, 0.5929598971107545, 0.6030100648583945, 0.6130602326060344, 0.6231104003536743, 0.6331605681013142, 0.643210735848954, 0.653260903596594, 0.6633110713442338, 0.6733612390918737, 0.6834114068395136, 0.6934615745871536, 0.7035117423347935, 0.7135619100824334, 0.7236120778300733, 0.7336622455777132, 0.7437124133253531, 0.753762581072993, 0.763812748820633, 0.7738629165682729, 0.7839130843159128, 0.7939632520635527, 0.8040134198111926, 0.8140635875588325, 0.8241137553064724, 0.8341639230541122, 0.8442140908017521, 0.8542642585493921, 0.864314426297032, 0.8743645940446719, 0.8844147617923118, 0.8944649295399517, 0.9045150972875916, 0.9145652650352315, 0.9246154327828715, 0.9346656005305114, 0.9447157682781513, 0.9547659360257912, 0.9648161037734311, 0.974866271521071, 0.9849164392687109, 0.9949666070163508, 1.0050167747639907, 1.0150669425116305, 1.0251171102592704, 1.0351672780069103, 1.0452174457545502, 1.0552676135021903, 1.0653177812498302, 1.0753679489974701, 1.08541811674511, 1.09546828449275, 1.1055184522403898, 1.1155686199880297, 1.1256187877356696, 1.1356689554833095, 1.1457191232309494, 1.1557692909785893, 1.1658194587262292, 1.175869626473869, 1.185919794221509, 1.1959699619691488, 1.206020129716789, 1.2160702974644289, 1.2261204652120687, 1.2361706329597086, 1.2462208007073485, 1.2562709684549884, 1.2663211362026283, 1.2763713039502682, 1.286421471697908, 1.296471639445548, 1.306521807193188, 1.3165719749408278, 1.3266221426884677, 1.3366723104361076, 1.3467224781837475, 1.3567726459313874, 1.3668228136790272, 1.3768729814266674, 1.3869231491743073, 1.3969733169219472, 1.407023484669587, 1.417073652417227, 1.4271238201648668, 1.4371739879125067, 1.4472241556601466, 1.4572743234077865, 1.4673244911554264, 1.4773746589030663, 1.4874248266507062, 1.497474994398346, 1.507525162145986, 1.5175753298936259, 1.527625497641266, 1.5376756653889059, 1.5477258331365458, 1.5577760008841857, 1.5678261686318256, 1.5778763363794654, 1.5879265041271053, 1.5979766718747452, 1.6080268396223851, 1.618077007370025, 1.628127175117665, 1.6381773428653048, 1.6482275106129447, 1.6582776783605846, 1.6683278461082245, 1.6783780138558644, 1.6884281816035043, 1.6984783493511444, 1.7085285170987843, 1.7185786848464242, 1.728628852594064, 1.738679020341704, 1.7487291880893439, 1.7587793558369837, 1.7688295235846236, 1.7788796913322635, 1.7889298590799034, 1.7989800268275433, 1.8090301945751832, 1.819080362322823, 1.829130530070463, 1.839180697818103, 1.849230865565743, 1.859281033313383, 1.8693312010610228, 1.8793813688086627, 1.8894315365563026, 1.8994817043039425, 1.9095318720515824, 1.9195820397992223, 1.9296322075468622, 1.939682375294502, 1.949732543042142, 1.9597827107897818, 1.9698328785374217, 1.9798830462850616, 1.9899332140327015, 1.9999833817803414, 2.0100335495279813, 2.020083717275621, 2.030133885023261, 2.040184052770901, 2.050234220518541, 2.0602843882661808, 2.0703345560138207, 2.0803847237614606, 2.0904348915091004, 2.100485059256741, 2.1105352270043807, 2.1205853947520206, 2.1306355624996605, 2.1406857302473004, 2.1507358979949402, 2.16078606574258, 2.17083623349022, 2.18088640123786, 2.1909365689855, 2.2009867367331397, 2.2110369044807796, 2.2210870722284195, 2.2311372399760594, 2.2411874077236993, 2.251237575471339, 2.261287743218979, 2.271337910966619, 2.281388078714259, 2.2914382464618988, 2.3014884142095386, 2.3115385819571785, 2.3215887497048184, 2.3316389174524583, 2.341689085200098, 2.351739252947738, 2.361789420695378, 2.371839588443018, 2.381889756190658, 2.3919399239382977, 2.4019900916859376, 2.412040259433578, 2.422090427181218, 2.4321405949288577, 2.4421907626764976, 2.4522409304241375, 2.4622910981717774, 2.4723412659194173, 2.482391433667057, 2.492441601414697, 2.502491769162337, 2.512541936909977, 2.5225921046576167, 2.5326422724052566, 2.5426924401528965, 2.5527426079005364, 2.5627927756481763, 2.572842943395816, 2.582893111143456, 2.592943278891096, 2.602993446638736, 2.613043614386376, 2.6230937821340157, 2.6331439498816556, 2.6431941176292955, 2.6532442853769354, 2.6632944531245752, 2.673344620872215, 2.683394788619855, 2.693444956367495, 2.703495124115135, 2.7135452918627747, 2.7235954596104146, 2.7336456273580545, 2.743695795105695, 2.7537459628533347, 2.7637961306009746, 2.7738462983486145, 2.7838964660962544, 2.7939466338438943, 2.803996801591534, 2.814046969339174, 2.824097137086814, 2.834147304834454, 2.8441974725820938, 2.8542476403297337, 2.8642978080773736, 2.8743479758250134, 2.8843981435726533, 2.8944483113202932, 2.904498479067933, 2.914548646815573, 2.924598814563213, 2.934648982310853, 2.9446991500584927, 2.9547493178061326, 2.9647994855537725, 2.9748496533014124, 2.9848998210490523, 2.994949988796692, 3.005000156544332, 3.015050324291972, 3.025100492039612, 3.0351506597872517, 3.0452008275348916, 3.055250995282532, 3.065301163030172, 3.0753513307778118, 3.0854014985254516, 3.0954516662730915, 3.1055018340207314, 3.1155520017683713, 3.125602169516011, 3.135652337263651, 3.145702505011291, 3.155752672758931, 3.165802840506571, 3.1758530082542107, 3.1859031760018506, 3.1959533437494905, 3.2060035114971304, 3.2160536792447703, 3.22610384699241, 3.23615401474005, 3.24620418248769, 3.25625435023533, 3.2663045179829697, 3.2763546857306096, 3.2864048534782495, 3.2964550212258894, 3.3065051889735293, 3.316555356721169, 3.326605524468809, 3.336655692216449, 3.346705859964089, 3.3567560277117288, 3.3668061954593687, 3.3768563632070085, 3.386906530954649, 3.396956698702289, 3.4070068664499287, 3.4170570341975686, 3.4271072019452085, 3.4371573696928484, 3.4472075374404882, 3.457257705188128, 3.467307872935768, 3.477358040683408, 3.487408208431048, 3.4974583761786877, 3.5075085439263276, 3.5175587116739675, 3.5276088794216074, 3.5376590471692473, 3.547709214916887, 3.557759382664527, 3.567809550412167, 3.577859718159807, 3.5879098859074467, 3.5979600536550866, 3.6080102214027265, 3.6180603891503664, 3.6281105568980063, 3.638160724645646, 3.648210892393286, 3.658261060140926, 3.668311227888566, 3.678361395636206, 3.6884115633838457, 3.698461731131486, 3.708511898879126, 3.718562066626766, 3.7286122343744057, 3.7386624021220456, 3.7487125698696855, 3.7587627376173254, 3.7688129053649653, 3.778863073112605, 3.788913240860245, 3.798963408607885, 3.809013576355525, 3.8190637441031647, 3.8291139118508046, 3.8391640795984445, 3.8492142473460844, 3.8592644150937243, 3.869314582841364]
    y = [-7.63171019399352E-5, -7.726616126939168E-5, -7.823120898145698E-5, -7.921258855575342E-5, -8.02106525368357E-5, -8.122576282119682E-5, -8.22582909537994E-5, -8.330861843671011E-5, -8.437713704878414E-5, -8.54642491774595E-5, -8.65703681631376E-5, -8.769591865712172E-5, -8.88413369926961E-5, -9.000707157074273E-5, -9.119358326066036E-5, -9.240134581656393E-5, -9.363084631017741E-5, -9.488258558012106E-5, -9.615707870036738E-5, -9.745485546608842E-5, -9.877646090000931E-5, -1.0012245577884673E-4, -1.0149341718133039E-4, -1.0288993905864044E-4, -1.0431263282841474E-4, -1.0576212799367579E-4, -1.072390727871979E-4, -1.0874413484340074E-4, -1.1027800189831511E-4, -1.118413825198242E-4, -1.1343500686868E-4, -1.1505962749317069E-4, -1.1671602015740092E-4, -1.1840498470658002E-4, -1.2012734597009869E-4, -1.2188395470474699E-4, -1.236756885799085E-4, -1.2550345320745746E-4, -1.273681832177442E-4, -1.2927084338520685E-4, -1.312124298051849E-4, -1.331939711256798E-4, -1.352165298358575E-4, -1.3728120361575674E-4, -1.393891267490561E-4, -1.4154147160356817E-4, -1.4373945018257854E-4, -1.4598431575134542E-4, -1.4827736454207403E-4, -1.5061993754353868E-4, -1.5301342237834353E-4, -1.5545925527389105E-4, -1.5795892313229606E-4, -1.605139657050392E-4, -1.6312597787809368E-4, -1.6579661207471525E-4, -1.685275807829234E-4, -1.7132065921371848E-4, -1.7417768810052824E-4, -1.7710057664638518E-4, -1.8009130562832393E-4, -1.8315193067014988E-4, -1.8628458569172948E-4, -1.894914865487423E-4, -1.9277493487225827E-4, -1.9613732212314032E-4, -1.9958113387400856E-4, -2.031089543338979E-4, -2.067234711319681E-4, -2.1042748037641005E-4, -2.1422389200832694E-4, -2.18115735469889E-4, -2.2210616570784818E-4, -2.2619846953527204E-4, -2.3039607237848906E-4, -2.3470254543209255E-4, -2.3912161325568502E-4, -2.4365716183938693E-4, -2.483132471749102E-4, -2.530941043669968E-4, -2.580041573257185E-4, -2.630480290819655E-4, -2.682305527728916E-4, -2.735567833473242E-4, -2.7903201004626484E-4, -2.8466176971689603E-4, -2.9045186102601893E-4, -2.964083596411444E-4, -3.0253763445745556E-4, -3.0884636495198274E-4, -3.1534155975747617E-4, -3.2203057655226964E-4, -3.2892114337636875E-4, -3.360213814907845E-4, -3.4333982990683343E-4, -3.50885471730417E-4, -3.5866776247192784E-4, -3.6669666049181924E-4, -3.749826597699164E-4, -3.83536825196624E-4, -3.923708306163365E-4, -4.014969998640376E-4, -4.109283510686252E-4, -4.2067864452054876E-4, -4.3076243443184847E-4, -4.4119512495597083E-4, -4.519930308620316E-4, -4.6317344331417475E-4, -4.7475470124198036E-4, -4.8675626885003737E-4, -4.99198819868806E-4, -5.12104329216691E-4, -5.254961728210687E-4, -5.393992364234672E-4, -5.538400342967734E-4, -5.688468388997595E-4, -5.844498226229038E-4, -6.006812129052779E-4, -6.17575462164325E-4, -6.351694341443317E-4, -6.535026084905236E-4, -6.726173055745843E-4, -6.925589338526513E-4, -7.133762623140705E-4, -7.351217209168029E-4, -7.57851732265951E-4, -7.816270782195312E-4, -8.065133055932345E-4, -8.325811756782181E-4, -8.599071629344034E-4, -8.885740089405484E-4, -9.186713385233241E-4, -9.502963459476145E-4, -9.835545601724338E-4, -0.0010185606994410313, -0.001055439626976122, -0.0010943274212605089, -0.0011353725763789911, -0.0011787373502126146, -0.0012245992809847355, -0.0012731528957905698, -0.0013246116384250939, -0.0013792100481253573, -0.0014372062258772482, -0.0014988846308631803, -0.0015645592565835867, -0.0016345772444000873, -0.0017093230019246983, -0.0017892229051553858, -0.0018747506768284077, -0.0019664335495909392, -0.002064859341775743, -0.0021706845964154166, -0.0022846439614201573, -0.0024075610214766984, -0.0025403608313143413, -0.0026840844468957307, -0.002839905807468587, -0.003009151389268003, -0.0031933231334509106, -0.0033941252495064194, -0.0036134956145531406, -0.003853642632904182, -0.004117088594265283, -0.004406720779107136, -0.00472585181334978, -0.005078291079921196, -0.005468429361334028, -0.0059013393253941684, -0.006382894985415029, -0.006919913875677851, -0.00752032638656086, -0.00819337749722839, -0.008949867005056579, -0.009802435226430404, -0.010765901924197121, -0.011857666698124682, -0.013098178889161854, -0.014511483555277248, -0.016125846177657107, -0.01797445058813737, -0.020096149038499572, -0.02253621511624776, -0.025347000612885036, -0.028588311966825902, -0.032327176564861004, -0.03663642461323955, -0.041591104348381085, -0.047261074008852905, -0.05369701117182237, -0.06090530240943591, -0.06880446997470063, -0.07715149956835904, -0.08542018483343272, -0.092605238166461, -0.09691637844732924, -0.09531958598101978, -0.08288736770489859, -0.05195392279111377, 0.008839002402717157, 0.11534989163198119, 0.2881073257546963, 0.5511682470425484, 0.9291377143812344, 1.442582448007854, 2.103242450764413, 2.9110381314737808, 3.8540002348732756, 4.910523188424698, 6.052347262162738, 7.24704635358864, 8.463299138453657, 9.669019329866067, 10.825245473050556, 11.89755618710376, 12.853401157615455, 13.66340639455548, 14.30295120811052, 14.752939511820042, 14.998332749865185, 15.024286465352935, 14.811999639396083, 14.33761317057658, 13.576302759360154, 12.51059569646428, 11.138928997766605, 9.480032503536712, 7.571514790251174, 5.464547988216373, 3.2178235671748143, 0.892851202716673, -1.4488715580796316, -3.7470814158567456, -5.943063834119788, -7.98035841895401, -9.807360116262883, -11.382445893551687, -12.6805073049308, -13.698106104216029, -14.45422350977166, -14.985476981502288, -15.337615193149754, -15.556775517713122, -15.683306434811886, -15.748992229638663, -15.776953012528539, -15.782990076906025, -15.777373326493224, -15.766500628438095, -15.754208530723718, -15.74271352245353, -15.733246435339668, -15.72646038782053, -15.722682076891124, -15.722057548991122, -15.724625423252755, -15.730335177174585, -15.739014988837264, -15.750281170250535, -15.76336762352471, -15.77683759131, -15.78812124811627, -15.792804970360551, -15.783592649511876, -15.748890774935129, -15.67107921247764, -15.524769491730583, -15.275738955255303, -14.881646709653687, -14.295720517467796, -13.473814535081042, -12.383440120365677, -11.01157488874951, -9.367990641589484, -7.483062010694573, -5.401890011407997, -3.177910760687768, -0.8683964445190722, 1.4675615011592824, 3.7695842864366362, 5.976614935007563, 8.02876890243947, 9.871033413266057, 11.458975091227236, 12.765590962958386, 13.786771651217782, 14.542120738804423, 15.069723469571144, 15.416900231739033, 15.630926202614297, 15.752725630311248, 15.81423452796406, 15.838496368174209, 15.841152224651571, 15.832304580283857, 15.818204571125166, 15.80256552035209, 15.787497622237414, 15.774134301975817, 15.763033830051176, 15.75442707163625, 15.748363113459021, 15.744786816553113, 15.74356809430518, 15.744491416258343, 15.747204382367883, 15.751114835289775, 15.7552159171743, 15.757807836130095, 15.756076395507343, 15.745488853661458, 15.718992653691792, 15.666076651880797, 15.571904539116918, 15.416956015097115, 15.17782357391981, 14.829769621542932, 14.351037770403488, 13.727766026880767, 12.957497218117997, 12.049808820879976, 11.024317398827742, 9.908155854313287, 8.736696340113538, 7.536414650932108, 6.342633286239041, 5.190274095447473, 4.112420964586379, 3.1386375211812005, 2.292742034292023, 1.5900085296666904, 1.034595011570889, 0.6186956081399196, 0.32448932941515074, 0.12841055513709262, 0.005948532576532765, -0.06474741966430977, -0.10110978180572082, -0.11592808901147202, -0.11799121217516083, -0.11302591250250474, -0.10459553626700761, -0.09482093201285946, -0.0849027965012142, -0.07547587217616544, -0.06683827576207149, -0.059094985957257797, -0.05224519392025402, -0.046234179045260684, -0.04098332067774137, -0.036406899523794015, -0.032421059458869694, -0.028948204935414258, -0.02591880353393633, -0.023271762934549016, -0.020954066889127498, -0.018920064011623106, -0.01713063034815334, -0.015552324961288022, -0.014156598673342768, -0.012919082424381431, -0.011818963033505766, -0.010838444282651096, -0.009962286453051424, -0.00917741552704089, -0.008472592918905172, -0.00783813705786642, -0.0072656889774069195, -0.006748015023818612, -0.0062788407519660605, -0.005852710962571798, -0.005464871625017467, -0.005111170115739669, -0.0047879707885553225, -0.004492083388696962, -0.004220702237900883, -0.003971354464695563, -0.003741855842502798, -0.003530273037622732, -0.0033348912677434975, -0.0031541865362092938, -0.0029868017437413605, -0.002831526092525859, -0.002687277291567926, -0.002553086150381339, -0.0024280832131247676]


    Plot2.plot_xy(x,y,describe='b-')

    Plot2.show()