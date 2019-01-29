#include "position_library.hpp"

// for mandelbrot
// ./bin/main zoom -0.86351690793367237879098526735500363171122900926623390235656859029095985817479106667893417017331775248894204667404104759173049314094055150218014325200616883094376002965516933657614243657952728054695501187855097054392324039595415883494985229715906678884870521543681491308761687017203355756095815611887401303438035 0.24770085085542684897920154941114532978571652912585207591199032605489162434475579901621342900504326332001572471388836875257693078071821918832702805395251556576917743455093070180103998083138219966104076957094394557391349705788109482159372116384541942314989586824711647398290030452624776670470371203410076798241659 1.05924e-306 14000
// ./bin/main zoom -0.863516907933672378790985267355003631711229009266233902356568590290959858174791066678934170173317752488942046674041047591730493140940551502180143252006168830943760029655169336576142436579527280546955011878550970543923240395954158834949852297159066788848705215436812860635332602278160099751996136063974639 0.247700850855426848979201549411145329785716529125852075911990326054891624344755799016213429005043263320015724713888368752576930780718219188327028053952515565769177434550930701801039980831382199661040769570943945573913497057881094821593721163845419423149895868247117015536061719196690131056301225176190768 1.05269e-298 14000

namespace position_library {
    std::vector<Position> mandelbrot = {
        // 0) Very deep
        {
            550000,
            "1e-1086",
            "-1.768610493014677074503175653270226520239677907588665494837677257575304640978808475274635707362464044253014370289948538552508877464736415873052958422861932774670165994201643419934807500290056179906392909880374230601661671965436663874506006355684166693059189687544326482526337453326360163639772818993753021740632937840115380957766425092940720439911920812397880443241274616212526380871555846532502156439892026352831619587768336768186345867565251889103622267866223055366872757385322485553606302984011695749730200727740242949661790906981449438923948817795927101980894917081591610562406554244675206099799522186446427884314773626993347929810277790888202019035845973880637832335294368222957931354735878969938534303074032237618397187328436715391758039680667871461788151793412286894565873237610467572174105629653438005433391873958639508124883860426968801537270756998560434335574379853659221182422319763412022530545421664765603500398209444536908432136868648907188939238968853841659746686717617541828417173199448336773447645561102873230388632254317334566170379314718589610910079079436175134148945650553411",
            "0.001266613503868717702066411192242601576193940560471409817185010171762524792588903616691501346028502452530417599269384116816237002586460261272462170615382790262110756215389780859682964779212455295242650488799024701023353984576434859496345393442867544784349509799966996827374525729583822627564832207860235000491856039278975203253540119195661182532106440194050352510825207428197675168479460252154208762204074041030502712772770772439567249008997886131809082319952112293668096363959700371035596685905429248221153089843201890985651976151989928496969024027810874574434857210174914227391125217932725188214796457327981771026544613194033736960542354861910879704489564999937473456191049937984461971508132204319961501958583967780282332682705656745932852354591955251196335374396883193221988201865629549575259395090238463522557833659758739138043696167112257784649600743807944457388512639475417466113111928274012056049434349358618953361438127758918999578120953045365596358997480091072548929426951083179599722132179281125708039705266879359303320165515458347343055671220673027817611220892213570374041225632346"
        },
        // 1) Deep
        {
            14000,
            "1e-298",
            "-0.863516907933672378790985267355003631711229009266233902356568590290959858174791066678934170173317752488942046674041047591730493140940551502180143252006168830943760029655169336576142436579527280546955011878550970543923240395954158834949852297159066788848705215436812860635332602278160099751996136063974639",
            "0.247700850855426848979201549411145329785716529125852075911990326054891624344755799016213429005043263320015724713888368752576930780718219188327028053952515565769177434550930701801039980831382199661040769570943945573913497057881094821593721163845419423149895868247117015536061719196690131056301225176190768"
        },
        // 2) Mosaic
        {
            20000,
            "3.47252e-47",
            "0.372137738770323258373356630885867793129508737859268",
            "-0.090398245434178161692952411151009819302665482561413"
        },
        // 3) Hard
        {
            45000,
            "6e-141",
            "-1.47994622332507888020258065344256383359082887482853327232891946750450142804155145810212315771521365103554594354207816734895388578734190261250998672",
            "0.00090139732902035398019779186619717356625156681804510241106763038648869228718904914562158443602693421876352757729063180945479661811076742458032279"
        }
    };
};