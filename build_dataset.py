"""
build_dataset.py
-----------------
Builds data/train.jsonl, data/val.jsonl, data/test.jsonl for Task 5.

SOURCING NOTE (read this before trusting the data):
Every "case" below is a REAL, named grassroots innovator documented by the
National Innovation Foundation - India (NIF), the Honey Bee Network, or
Wikipedia/mainstream news coverage of NIF/Padma Shri awardees. The facts
(name, state, problem, device) were gathered via web search in this session
from nif.org.in, en.wikipedia.org, and secondary sources (see SOURCES.md).

The `problem` / `solution` fields and the paragraph text are ORIGINAL
WRITING (paraphrased, not copied) based on those facts — they are NOT
copy-pasted from NIF pages, to respect copyright. This means the labels are
"auto-generated" in the sense the task allows ("bootstrap labels with an
API LLM"), authored here by Claude directly from sourced facts, and have NOT
been separately hand-verified by a second human. Treat this as a starter
corpus: spot-check a sample against SOURCES.md before treating it as gold.

Each case is expanded into 4-5 paragraph *paraphrase variants* (different
phrasing/length/order of the same true facts) — this is a deliberate
data-augmentation step to reach a usable row count (~500+) from a
larger number of independently-sourced real cases (129). Because all
variants of one case share one underlying story, we split by CASE, not by
row, so no case's variants leak across train/val/test.
"""
import json
import random
import os

random.seed(42)

# ---------------------------------------------------------------------------
# 129 real, sourced grassroots-innovation cases (37 from the original pass,
# plus 92 more added from NIF's 11th and 10th National Biennial Award Books
# to scale train.jsonl up toward ~400 rows).
# problem / solution are each written as 1-2 original sentences.
# ---------------------------------------------------------------------------
CASES = [
    dict(id="remya_jose", name="Remya Jose", place="Palakkad, Kerala",
         problem="A schoolgirl in rural Kerala had to spend hours each day hand-washing clothes for her family while also managing her studies and a sick parent, with no washing machine affordable or available.",
         solution="She designed a low-cost pedal-powered washing machine that uses foot power instead of electricity, cutting washing time and effort for households without power or money for an electric machine."),
    dict(id="appachan", name="Appachan", place="Kerala",
         problem="Workers who climb tall coconut and palm trees to harvest nuts or maintain electric poles faced a slow, physically risky climb with no safety support.",
         solution="He built a simple mechanical tree-climbing device that lets a person ascend tall trunks safely and quickly, reducing the risk of falls during harvesting or pole maintenance."),
    dict(id="saidullah", name="Mohammad Saidullah", place="Bihar",
         problem="People in flood-prone parts of Bihar had no easy way to commute once roads and low areas were submerged, cutting them off during the monsoon.",
         solution="He converted an ordinary bicycle into an amphibious version that can also move across water, letting people cross flooded roads, ponds, and small water bodies during floods."),
    dict(id="dhanjibhai", name="Dhanjibhai Kerai", place="Gujarat",
         problem="As a physically challenged person himself, Dhanjibhai found that standard scooters were unusable for people with limited mobility, leaving them with few affordable transport options.",
         solution="He modified an existing scooter with hand-operated controls so that physically challenged riders can operate it safely, giving them independent mobility."),
    dict(id="jagani_sprayer", name="Mansukhbhai Jagani", place="Gujarat",
         problem="Small farmers spraying pesticide by hand found it slow, tiring, and imprecise, especially over larger fields.",
         solution="He built a bicycle-mounted pesticide sprayer where pedalling powers the spray pump, letting one person cover a field faster and with less physical strain than manual spraying."),
    dict(id="jayanti_patel", name="Jayanti J. Patel", place="Gujarat",
         problem="Standard single-speed bicycles used by rural workers struggled on uneven terrain and inclines common in the countryside.",
         solution="He added a gear mechanism to an ordinary bicycle, making it easier to pedal uphill and over rough rural roads."),
    dict(id="dodhi_pathak", name="Dodhi Pathak", place="Assam",
         problem="Conventional steel bicycle frames were costly for poor rural users in Assam, where bamboo is abundant and cheap.",
         solution="He built a bicycle with a bamboo frame, using a locally available low-cost material in place of steel to bring down the price of the cycle."),
    dict(id="jagtap_spray", name="Subhas Vasantrao Jagtap", place="Maharashtra",
         problem="Farmers spraying crops manually struggled to carry pump equipment across large plots without help.",
         solution="He mounted a spray pump on a bicycle so the pedalling motion drives the pump, allowing one farmer to spray a field while moving through it."),
    dict(id="kamruddin_bike", name="Md. Kamruddin", place="Rajasthan",
         problem="Rural households needed one vehicle to serve several everyday jobs — carrying loads, pumping water, or farm chores — but could not afford separate machines for each task.",
         solution="He modified a bicycle into a multipurpose unit that can be adapted to different rural chores beyond simple transport, reducing the need for multiple separate machines."),
    dict(id="rathore_pump", name="Vikram Rathore", place="Andhra Pradesh",
         problem="Small farmers without electricity connections had no affordable way to lift water from shallow sources to irrigate small plots.",
         solution="He built a bicycle-operated water pump, so pedalling draws water for irrigation without needing electricity or a diesel engine."),
    dict(id="gayen_pump", name="Nasiruddin Gayen", place="West Bengal",
         problem="Farmers needing to move a water pump between different small, scattered plots found fixed pump installations inconvenient.",
         solution="He built a bicycle-based portable pump that can be wheeled to wherever water is needed and operated by pedalling."),
    dict(id="kanak_das", name="Kanak Das", place="Assam",
         problem="Riders on rough Assamese village roads had no way to get extra propulsion from a bicycle's motion beyond normal pedalling.",
         solution="He designed a rider-induced bicycle mechanism that lets the rider's own body movement add extra drive to the cycle."),
    dict(id="marvaniya_carrot", name="Vallabhbhai Marvaniya", place="Junagadh, Gujarat",
         problem="Farmers in Gujarat lacked a carrot variety suited to local soil and climate that was also highly nutritious.",
         solution="He developed and popularised the Madhuban Gajar carrot variety, a locally adapted, nutrient-rich carrot that farmers in the region could grow successfully."),
    dict(id="cv_raju_toys", name="C. V. Raju", place="Andhra Pradesh",
         problem="A traditional wooden and lacquer toy-making craft in his region was declining as artisans struggled to sustain it commercially.",
         solution="He worked to revive and promote the traditional toy-making craft, helping it gain recognition (including a UNESCO Seal of Excellence for Handicrafts) and reach new markets."),
    dict(id="dipak_sardar", name="Dipak Sardar", place="West Bengal",
         problem="Producing sola (shola) pith wood sheets by hand for craft and decoration was slow and labour-intensive.",
         solution="He built a machine that mechanises the cutting of sola wood into sheets, speeding up production for this traditional craft material."),
    dict(id="suthar_groundnut", name="Kishan Lal Suthar", place="Rajasthan",
         problem="Farmers manually decorticating and grading groundnut after harvest spent significant time and labour on the process.",
         solution="He built a tractor-operated groundnut decorticator-cum-grader that mechanises shelling and sorting groundnut in one pass."),
    dict(id="maurya_guava", name="Ram Vilas Maurya", place="Uttar Pradesh",
         problem="Farmers wanted a higher-quality, better-yielding guava variety suited to local growing conditions.",
         solution="He developed the 'G-Vilas Pasand' improved guava variety, giving farmers a better-performing option to cultivate."),
    dict(id="aniyamma_cashew", name="Aniyamma Baby", place="Kerala",
         problem="Propagating cashew plants reliably from cuttings was difficult using existing rooting methods, limiting how easily good cashew varieties could be multiplied.",
         solution="She developed a multiple-rooting propagation method for cashew that improves how reliably new cashew plants can be grown from cuttings."),
    dict(id="chopade_embossing", name="Ravi Ganpat Chopade", place="Maharashtra",
         problem="Embossing designs onto material by hand, one angle at a time, was slow for small workshops doing decorative or packaging work.",
         solution="He built a six-axis rotating-head golden embossing machine that automates embossing from multiple angles, speeding up the process."),
    dict(id="majhi_transplanter", name="Sadasibo Majhi", place="Odisha",
         problem="Transplanting paddy seedlings by hand is slow, back-breaking work that ties up labour at a critical point in the season.",
         solution="He built a manual paddy transplanter that lets a farmer transplant seedlings faster and with less strain than transplanting entirely by hand."),
    dict(id="yusuf_groundnut_digger", name="Yusuf Khan", place="Gujarat",
         problem="After harvesting groundnut, up to a fifth of the pods were left buried in the soil and recovering them by hand was slow, costly, and labour-scarce.",
         solution="He built a tractor-operated groundnut digger that mechanically digs, sieves, and collects leftover pods from the soil, recovering yield that manual labour would otherwise miss."),
    dict(id="parikh_cauliflower", name="Jagdish Prasad Parikh", place="Rajasthan",
         problem="Farmers needed a cauliflower variety that could better withstand local pests and climate stress, since standard varieties often failed under these conditions.",
         solution="He developed a pest- and climate-resistant cauliflower variety that performs more reliably for farmers in tough growing conditions."),
    dict(id="pachar_carrot", name="Santosh Pachar", place="Rajasthan",
         problem="Local farmers lacked a carrot variety well suited to their specific growing region.",
         solution="She developed a new carrot variety adapted to local conditions, giving farmers in the area a better-performing crop option."),
    dict(id="kumawat_thresher", name="Madanlal Kumawat", place="Rajasthan",
         problem="Farmers growing several different crops needed separate threshing equipment for each one, which was expensive and inefficient for small landholders.",
         solution="He built a multi-crop thresher that can process more than one type of crop, reducing the need to buy separate machines for each."),
    dict(id="ola_heat_recovery", name="Subhash Ola", place="Rajasthan",
         problem="Small-scale processing setups were wasting the heat given off during condensation, adding unnecessary fuel cost.",
         solution="He built a condensate and heat recovery system that captures and reuses this waste heat, cutting fuel costs for the process."),
    dict(id="dahiya_gasifier", name="Rai Singh Dahiya", place="Rajasthan",
         problem="Rural households and small farms needed an affordable source of fuel gas but had limited access to conventional cooking or power fuel.",
         solution="He built a biomass gasifier system that converts crop residue and organic waste into usable gas fuel."),
    dict(id="sundaram_verma", name="Sundaram Verma", place="Rajasthan",
         problem="Farmers in his arid region struggled to grow crops reliably given scarce and unpredictable rainfall.",
         solution="He carried out decades of on-farm research into dryland cropping and water-conservation techniques, developing practices that help farmers grow crops with very little water, and went on to scout and mentor several other grassroots innovators for NIF."),
    dict(id="khass_planter", name="Indrajit Singh Khass", place="India (region not confirmed)",
         problem="Planting turmeric and ginger by hand at consistent spacing was slow and required significant manual labour.",
         solution="He built a tractor-mounted turmeric and ginger planter with adjustable row spacing that opens the furrow, meters the seed, and places it automatically."),
    dict(id="mohanbhai_thresher", name="Mohanbhai", place="Sabarkantha, Gujarat",
         problem="Groundnut farmers faced a shortage of labour at harvest time, which delayed collection of dried crop and caused yield losses.",
         solution="He built a tractor-mounted, PTO-powered mobile thresher with separate chambers for groundnut pods and stalks, mechanising collection and threshing together."),
    dict(id="nilesh_pankaj_thresher", name="Nileshbhai and Pankaj", place="Gujarat",
         problem="Collecting harvested and dried groundnut stalks from the field by hand delayed processing and caused losses when labour was scarce.",
         solution="They built a modified tractor-mounted mobile automatic thresher that moves through the field collecting, threshing, and separating groundnut pods and stalks on its own."),
    dict(id="nadakattin_tamarind", name="Abdul Khader Nadakattin", place="Dharwad, Karnataka",
         problem="Separating tamarind seeds from the pulp by hand was slow, tedious work for processors and farmers.",
         solution="He built a device that mechanically separates tamarind seeds from pulp, speeding up processing for a crop with limited mechanisation."),
    dict(id="nadakattin_drill", name="Abdul Khader Nadakattin", place="Dharwad, Karnataka",
         problem="Small farmers sowing seed and applying fertiliser separately spent extra time and often placed fertiliser inconsistently relative to the seed.",
         solution="He built a seed-cum-fertiliser drill, along with an automatic sugarcane sowing driller and a wheel tiller, so farmers can sow seed and place fertiliser together in one pass over the field."),
    dict(id="bharali_deseeder", name="Uddhab Bharali", place="Lakhimpur, Assam",
         problem="Deseeding pomegranates by hand for juice or processing was slow and inefficient for small producers.",
         solution="He built a low-cost pomegranate deseeder, one of more than 160 low-cost devices he has invented for everyday rural and small-industry problems."),
    dict(id="hariman_apple", name="Hariman Sharma", place="Himachal Pradesh",
         problem="Apple cultivation was traditionally seen as limited to cold, high-altitude regions, leaving warmer low-altitude farmers unable to grow the crop.",
         solution="He developed the HRMN-99 apple variety, which can flower and fruit even in warm conditions up to around 45°C, letting farmers in warmer regions grow apples for the first time."),
    dict(id="manipur_poultry_herbal", name="Oinam Ibetombi Devi, Sarangthen Dasumati Devi and Nameirakpam Sanahambi Devi", place="Nambol, Manipur",
         problem="Poultry keepers in the area had no affordable local treatment for coccidiosis, a common intestinal infection that hurts bird health and growth.",
         solution="Three women from the village developed a herbal feed formulation made from a local plant that, fed to birds for 5-7 days, reduced intestinal damage and improved weight gain compared with untreated birds, performing comparably to a standard drug."),
    dict(id="kundu_herbal", name="Ishwar Singh Kundu", place="Kaithal, Haryana",
         problem="Farmers relying on separate chemical products for soil fertility and pest control faced higher costs and more chemical use.",
         solution="He developed a multi-utility herbal formulation that works as a bio-fertiliser, soil enhancer, and pest-control treatment in one product, cutting the need for multiple separate chemical inputs."),
    dict(id="prajapati_mitticool", name="Mansukhbhai Prajapati", place="Wankaner, Gujarat",
         problem="Many poor rural households in Gujarat had no electricity connection or could not afford a refrigerator, leaving them without a way to keep food and drinking water cool.",
         solution="A potter by trade, he designed 'Mitticool,' a refrigerator made entirely of unglazed clay that cools its contents through natural evaporation and needs no electricity to run."),

    # -----------------------------------------------------------------------
    # 92 additional cases sourced from NIF's 11th (2023) and 10th (2019)
    # National Biennial Award Books (nif.org.in/upload/11th-Final-Award-Book-
    # April-2023.pdf and nif.org.in/upload/10th-National-Award-Book-2019-new.pdf).
    # Same rules as above: facts are from NIF's own award write-ups; the
    # problem/solution wording below is original, paraphrased by Claude, not
    # copied from NIF's text. See SOURCES.md for the per-case source mapping.
    # -----------------------------------------------------------------------
    dict(id="indrajit_khass_stump", name="Indrajit Singh Khas", place="Aurangabad, Maharashtra",
         problem="A farmer friend had planted paper-wood (subabul) trees, and after harvest their stumps were left in the ground; the only removal option, hiring a JCB excavator, cost far more than the farmer could afford for his land.",
         solution="He built a tractor-mounted stump remover that uses the tractor's hydraulic power to grip and pull tree stumps out of the ground, offering a cheaper alternative to renting an excavator."),
    dict(id="karekar_turmeric", name="Sachin Kamlakar Karekar", place="Ratnagiri, Maharashtra",
         problem="Turmeric farmers in his area lacked a variety that combined early maturity, high yield, and resistance to rhizome rot disease.",
         solution="Through years of selecting the best-performing plants from a local variety, he developed the SK-4 turmeric variety, which matures early, yields more, and tolerates rhizome rot better than standard varieties."),
    dict(id="shishpal_herbal", name="Shishpal Singh", place="Meerut, Uttar Pradesh",
         problem="Livestock owners had no reliable way to help cows and buffaloes expel the placenta and complete uterine recovery quickly after calving, risking the animal's health.",
         solution="He developed a herbal medication that, when administered after calving, speeds up expulsion of the placenta and involution of the uterus in the animal."),
    dict(id="biju_coconut", name="Biju Narayanan", place="Idukki, Kerala",
         problem="Breaking open dry coconuts by hand for processing into copra, oil, or powder was unsafe, tiring, and slow work.",
         solution="He built a portable device that splits a dry coconut in seconds using a falling blade and cam mechanism, while also filtering and collecting the coconut water, and can run manually or with an electric motor."),
    dict(id="singh_cauliflower", name="Rajmani Singh and Mukesh Kumar Singh", place="Vaishali, Bihar",
         problem="Farmers growing traditional cauliflower cultivars in their district lacked an early-maturing, high-yielding variety with good curd quality.",
         solution="Through years of selecting and multiplying seed from a traditional cauliflower variety, the brothers developed Sonali-45, an early, high-yielding cauliflower with compact white curds, and went on to sell the seed commercially."),
    dict(id="vishnu_mastitis", name="Vishnu Kumar Sharma", place="Jaipur, Rajasthan",
         problem="Dairy farmers had no simple treatment for mastitis, an udder infection that causes pain, swelling, and reduced milk quality in cows.",
         solution="He developed a herbal formulation that, applied to the affected udder, gradually relieves the symptoms of mastitis."),
    dict(id="karappan_handloom", name="Karappan Venkatraman", place="Coimbatore, Tamil Nadu",
         problem="When a chikungunya outbreak left many local handloom weavers with severe joint pain, they could no longer operate looms that needed both legs and hands to run.",
         solution="He designed a handloom that can be woven using just one leg and no hands, letting weavers affected by joint pain keep working."),
    dict(id="shine_clove", name="Shine Joseph", place="Kozhikode, Kerala",
         problem="Separating clove buds from harvested stalks by hand required dozens of labourers to process a day's harvest, making it slow and costly.",
         solution="He built a motorised machine that rubs clusters through a spiral-bladed drum to separate stalks from buds, cutting the cost of processing from about Rs 12 a kilogram to under Rs 1."),
    dict(id="jalendra_ironbar", name="Jalendra Kumar", place="Nawada, Bihar",
         problem="Fabricators cutting and bending iron bars with a standard cut-off saw had to replace the abrasive wheel frequently and endure loud noise during the job.",
         solution="He built an electrically operated cutting-and-bending machine that shears iron bars quickly and quietly without the wear issues of an abrasive wheel."),
    dict(id="shafi_walnut", name="Mohd. Shafi Ahanger", place="Anantnag, Jammu & Kashmir",
         problem="Peeling green walnuts by hand right after harvest was slow, tiring, and damaged the skin of workers' hands.",
         solution="He built a manual-and-powered walnut peeling machine that shreds and peels walnuts on a conveyor system, processing hundreds of kilograms an hour."),
    dict(id="imkong_pineapple", name="Imkongsunep", place="Mokokchung, Nagaland",
         problem="Peeling pineapples by hand in his hilly district was a careful, time-consuming task.",
         solution="He built a small lathe-like machine that mounts and rotates a pineapple while a blade peels it in a few seconds, adjustable to different pineapple sizes."),
    dict(id="chhuanmawia_laddu", name="Chhuanmawia", place="Lunglei, Mizoram",
         problem="Making sesame-and-sugar laddus by hand meant rolling hot sugar mixture into balls with bare palms, often burning the skin, while spinning long weaving threads by hand was similarly slow.",
         solution="He built a laddu-making machine that presses the hot mixture into uniform balls without hand contact, and a separate large-scale spindle machine that looms twenty threads at once."),
    dict(id="yanglem_taro", name="Yanglem Brajamani Singh", place="Bishnupur, Manipur",
         problem="Harvesting and washing taro by hand with a spade took many labourers and days of work, driving up costs for farmers.",
         solution="He built a tractor-mounted taro harvester with an attached washer that digs up and cleans the crop, cutting a job that took a week down to about a day."),
    dict(id="suchil_soilcrusher", name="Suchil Teron", place="Kamrup, Assam",
         problem="Farmers preparing hard, clod-filled soil for planting had no efficient way to break up clods and level uneven wetland fields in one pass.",
         solution="He built a cultivator-mounted soil clod crusher and a tractor-mounted wetland leveller that break up soil clods, bury weeds for mulching, and level fields for planting."),
    dict(id="aminuddin_tarpaulin", name="Mohammad Aminuddin", place="Kolkata, West Bengal",
         problem="Tarpaulin shop shelters in narrow, low-lying bylanes collected rainwater during monsoon storms, becoming heavy, unstable, and prone to collapse or spilling water on customers.",
         solution="He built a tarpaulin shed with a built-in drain channel that carries collected rainwater off the roof and into the street drain, and has sold thousands of units to fellow shopkeepers."),
    dict(id="rajesh_polanga", name="Rajesh Sahoo", place="Puri, Odisha",
         problem="Extracting oil-rich polanga seeds from their hard shells was done by hand-hammering after days of sun-drying, limiting how much one labourer could process.",
         solution="He built an electric-motor decorticator that cracks polanga seed shells between rollers, more than tripling daily wages for the villagers who now use it, including several women."),
    dict(id="shinde_silkworm", name="Sunil Ajurnarav Shinde", place="Jalna, Maharashtra",
         problem="Folding and binding V-shaped silkworm breeding nets by hand between rearing cycles was slow, limiting workers to about 200 nets a day.",
         solution="He built a pipe-frame pressing device that folds and holds nets in place so they can be tied quickly, raising output to 300 nets an hour."),
    dict(id="lakkad_sorghum", name="Himatbhai Virjibhai Lakkad", place="Bhavnagar, Gujarat",
         problem="Farmers in his region needed a sorghum variety that performed well for both grain and fodder while tolerating drought.",
         solution="Through years of selecting a naturally superior sorghum plant, he developed the 'AA' dual-purpose, drought-tolerant sorghum variety that yields well for both grain and fodder."),
    dict(id="aji_pelletisation", name="Aji Thomas", place="Wayanad, Kerala",
         problem="Cultivating paddy in the hilly terrain of his district was difficult, and transporting rice from other areas added time and cost.",
         solution="He developed a table-top pelletisation system that grows paddy saplings in nutrient-rich pellets, increasing tiller count and straw yield while cutting seed requirements sharply."),
    dict(id="prusty_strawcutter", name="Ajaya Kumar Prusty", place="Dhenkanal, Odisha",
         problem="Cutting paddy straw to the right size for mushroom beds by hand was slow and needed more than two workers per bed.",
         solution="He built an adjustable straw cutter that trims paddy straw bundles to size quickly, and has sold over a thousand units across Odisha and Jharkhand."),
    dict(id="binoy_pepper", name="Binoy Sebastian", place="Idukki, Kerala",
         problem="Threshing pepper berries from their stalks by hand was slow and tedious for growers.",
         solution="He built a rotating-drum pepper thresher with spiral blades that rubs berries free of their stalks at high efficiency, validated by the local Krishi Vigyan Kendra."),
    dict(id="dhedhi_cultivator", name="Rajanikant Raghavjibhai Dhedhi", place="Morbi, Gujarat",
         problem="Farmers had to attach and detach separate cultivator and blade-harrow implements for ploughing and harrowing, wasting time between operations.",
         solution="He built a combined implement that carries both a cultivator and a blade harrow on a reversible mount, letting a farmer switch between the two operations without swapping equipment."),
    dict(id="dinesh_channel", name="Dinesh Chaudhary", place="Durg, Chhattisgarh",
         problem="A local farmer needed a way to dig planting channels for banana cultivation without the slow, labour-heavy process of doing it by hand.",
         solution="He built a tractor-operated channel-making machine that digs and moves soil with rotating blades, a design now also used for papaya cultivation."),
    dict(id="mohan_cuphanger", name="Mohana Kumar G S", place="Kollam, Kerala",
         problem="Cup hangers used to collect rubber latex during tapping were made by hand, so no two were alike and some failed under the weight of collected latex.",
         solution="He built a motorised machine that automatically produces uniform, latex-resistant cup hangers at a rate of 1,200 an hour."),
    dict(id="selichum_cardamom", name="Selichum Sangtam", place="Tuensang, Nagaland",
         problem="Farmers plucking and pruning cardamom with ordinary knives and sickles often injured the plants and capsules in the process.",
         solution="He designed curved plucking and dual-edged pruning tools that harvest cardamom capsules and clean the plants without damaging them."),
    dict(id="budheswar_coconut", name="Budheswar Pamey", place="Dhemaji, Assam",
         problem="Harvesting coconuts required climbing trees using ropes or bare feet, a risky task made harder by a shrinking pool of skilled tree climbers, and separating pupae from Eri silk cocoons by boiling damaged yarn quality.",
         solution="He built a handheld, finger-gripped tool on a bamboo pole that plucks selected coconuts from the ground without climbing, and a separate device that opens Eri cocoons without boiling, preserving yarn quality."),
    dict(id="harkumar_juice", name="Harkumar Goswami", place="Guwahati, Assam",
         problem="The local sugarcane juice market was informal and often unhygienic, giving customers little confidence in a healthier alternative to bottled soft drinks.",
         solution="He built a self-operated vending machine that dispenses hygienic sugarcane juice with card, UPI, and wallet payment options, designed to be set up as a public kiosk."),
    dict(id="gobin_arecanut", name="Gobin Sinha", place="Unakoti, Tripura",
         problem="Climbing areca nut trees using jute ropes was unsafe, since ropes could slip or snap under load.",
         solution="He designed a low-cost metal climbing device with grippers and footrests that clamps around the tree trunk, letting the user climb safely by stepping and locking the tool upward."),
    dict(id="chinnakannu_moulding", name="Chinnakannu Muthusami", place="Namakkal, Tamil Nadu",
         problem="Existing machines for moulding areca leaves into disposable plates had drawbacks that made them hard to maintain and physically demanding to operate, especially for women.",
         solution="He redesigned the leaf-moulding machine with a hydraulic mechanism and paired moulds, making it easier to operate and able to produce around 1,700 plates in an eight-hour day."),
    dict(id="punamchand_onion", name="Punamchand Patidar", place="Jhalawar, Rajasthan",
         problem="Local farmers needed a higher-yielding onion variety with good shelf life suited to their region.",
         solution="Through selection from a traditional variety, he developed the Kansi No.1 onion, a high-yielding, disease-free variety with good shelf life that he distributed through farmer networks."),
    dict(id="mousumi_rice", name="Mousumi Biswas", place="Murshidabad, West Bengal",
         problem="Farmers in her area lacked an aromatic rice variety that combined high yield with tolerance to blast disease.",
         solution="By crossing and selecting parent rice varieties over several years, she developed M-Jamini, a fine-grained, aromatic rice variety that yields well and resists blast disease."),
    dict(id="anand_chilli", name="Anand Kumar Patel (community representative)", place="Gandhinagar, Gujarat",
         problem="A farming community in Rupal village wanted to preserve and improve a traditional chilli variety suited for making chutney and sauce.",
         solution="Over five decades, the community selected and conserved a local chilli variety, now called Palli, prized for its bright colour, high yield, and suitability for processed chilli products."),
    dict(id="akhmabhai_anestrus", name="Akhmabhai Vagadia", place="Mahisagar, Gujarat",
         problem="Cows and buffaloes affected by anestrus fail to enter their reproductive cycle, hurting a farmer's ability to breed and maintain their herd.",
         solution="He developed a herbal medication that, when administered to affected animals, helps resume normal ovarian cycling and improves conception rates."),
    dict(id="baltej_flower", name="Baltej Singh Matharu", place="Ferozepur, Punjab",
         problem="Showering flower petals during religious and social occasions like Gurudwara ceremonies was traditionally done by hand, and existing electric flower-shower machines needed a power source and couldn't be carried everywhere.",
         solution="He built a lightweight, petrol-engine-powered flower shower machine that blows petals through a flexible pipe, portable enough to use anywhere."),
    dict(id="nisar_standingbar", name="Nisar Ahmad Itoo", place="Anantnag, Jammu & Kashmir",
         problem="Painters and masons working along walls relied on wooden angle brackets to raise a working platform, but wood was getting costlier and the brackets had limited height range.",
         solution="He built a height-adjustable metal standing bar, adjustable from 5 to 15 feet, that gives construction workers a stable platform and can be disassembled for storage."),
    dict(id="bajrang_wheat", name="Bajrang Lal", place="Nagaur, Rajasthan",
         problem="Farmers wanted a higher-yielding wheat variety with better grain quality and resistance to rust disease.",
         solution="Through pure-line selection from a commercial variety, he developed BLK-Balaji, a high-yielding, protein-rich wheat variety resistant to stem and leaf rust."),
    dict(id="hom_leafcurl", name="Hom Prasad Chauhan", place="South Sikkim, Sikkim",
         problem="Chilli farmers in his district had no effective treatment for leaf curl disease and the whitefly and thrips that spread it.",
         solution="He developed a fermented herbal preparation from local plants that reduces leaf curl disease and pest populations in chilli while boosting yield."),
    dict(id="asim_kangri", name="Asim Sikander Mir", place="Kulgam, Jammu & Kashmir",
         problem="The traditional Kashmiri kangri fire pot can spill hot embers and cause burns if it tilts or falls.",
         solution="As a school student, he designed a 'gyroscopic' kangri with an inner spinning pot that keeps its orientation and stays upright even if the outer pot tilts, preventing embers from spilling."),
    dict(id="sayen_lpg", name="Sayen Akhtar Sheikh", place="South Andaman, Andaman & Nicobar Islands",
         problem="Removing the plastic cap on an LPG cylinder nozzle required pulling a nylon thread clip and the cap at the same time, and the thread often cut the user's fingers.",
         solution="After seeing a family member get hurt, this school student built an opener that pulls the clip lock and lifts the cap in one motion."),
    dict(id="sumit_bamboo", name="Sumit Murari", place="Bokaro, Jharkhand",
         problem="Weaving bamboo baskets by hand let a weaver make only three or four baskets a day, keeping their earnings low.",
         solution="He designed a bamboo basket weaving machine to mechanise the process, aiming to boost weavers' output and income."),
    dict(id="tsering_seabuckthorn", name="Tsering Omphel", place="Leh, Ladakh",
         problem="Sea buckthorn's thorny branches make hand-harvesting its berries slow and painful, and many berries go unpicked because they're out of reach.",
         solution="He designed a low-cost harvesting device that plucks sea buckthorn berries without requiring bare-handed picking, reducing injury and fruit wastage."),
    dict(id="hema_stretcher", name="Hema Pradhan", place="West Sikkim, Sikkim",
         problem="Standard stretchers jolt patients uncomfortably when carried over rough, uneven ground outside hospitals.",
         solution="After watching a patient be carried over rough terrain, she proposed a stretcher fitted with a spring-based shock absorber to cushion the ride."),
    dict(id="arushi_bed", name="Arushi Tandon", place="Kolkata, West Bengal",
         problem="Her bedridden grandmother and the family struggled every time she needed to move between bed, wheelchair, and commode.",
         solution="She designed a bed with an integrated wheelchair that detaches easily and slides over a commode, eliminating repeated transfers for the patient."),
    dict(id="mauwang_lifejacket", name="Mauwang Wangham", place="Longding, Arunachal Pradesh",
         problem="Standard life jackets keep a person afloat but provide no way to breathe if they need oxygen in a drowning emergency.",
         solution="After watching a friend drown, he proposed a life jacket fitted with an attached oxygen mask for use in water emergencies."),
    dict(id="raghuvanshi_seeds", name="Prakash Singh Raghuvanshi", place="Varanasi, Uttar Pradesh",
         problem="Farmers lacked access to a range of reliable, improved plant varieties suited to their local growing conditions.",
         solution="Over decades, he developed and popularised the 'Kudrat Seeds' line of improved plant varieties, earning recognition as one of NIF's lifetime-achievement grassroots innovators."),
    dict(id="shyambir_transplanter", name="Shyambir Singh and Ved Prakash", place="Palwal, Haryana",
         problem="Transplanting paddy seedlings became increasingly expensive as labour wages rose and workers grew scarce during the peak season.",
         solution="The two friends designed an engine-powered paddy transplanter, later modified to run on tractor power, that transplants up to nine rows of seedlings at once."),
    dict(id="sandip_onion", name="Sandip Vishram Ghole", place="Pune, Maharashtra",
         problem="Onion growers needed a variety that resisted purple blotch disease while offering a longer shelf life than standard cultivars.",
         solution="Through ten years of selective breeding from a local onion cultivar, he developed the Sandip Pyaz variety, which tolerates purple blotch disease and keeps well in storage."),
    dict(id="anang_goggles", name="Anang Tadar", place="Papum Pare, Arunachal Pradesh",
         problem="Visually impaired people can detect obstacles at ground level with a cane, but had no way to sense obstacles at waist height and above while walking.",
         solution="As a student, he built smart goggles fitted with ultrasonic sensors that detect obstacles ahead and alert the wearer with vibrations, complementing a blind cane rather than replacing it."),
    dict(id="periyasami_coccidiosis", name="Periyasami Ramasami", place="Salem, Tamil Nadu",
         problem="Poultry keepers had no accessible treatment for coccidiosis, a parasitic infection that causes bloody diarrhoea and high mortality in birds.",
         solution="Drawing on herbal knowledge passed down from his elders, he developed a three-ingredient herbal medication that, given to infected birds for a few days, significantly reduces mortality from coccidiosis."),
    dict(id="saravanamuthu_bed", name="S. Saravanamuthu", place="Nagercoil, Tamil Nadu",
         problem="His bedridden wife needed help using the toilet, which was stressful and embarrassing for her.",
         solution="He built a battery-powered cot with an attached toilet pot that moves into position and flushes by remote control, letting her use it without assistance."),
    dict(id="sanjeev_cauliflower", name="Sanjeev Kumar", place="Vaishali, Bihar",
         problem="Farmers wanted an early-maturing cauliflower cultivar with bigger, more compact curds than the traditional varieties they were growing.",
         solution="Through selective breeding over several years, he developed the 'Sanjeev Selection' cauliflower, an early, high-yielding variety with compact white curds."),
    dict(id="vilat_anestrus", name="Vilat Yadav", place="Sitamarhi, Bihar",
         problem="Cows and buffaloes affected by anestrus fail to enter their breeding cycle, limiting a farmer's ability to grow their herd.",
         solution="He developed a herbal powder treatment that, mixed in water and given twice daily, brought most treated animals into oestrus within about two weeks in validation trials."),
    dict(id="maharaj_tillage", name="Maharaj Singh Lodhi", place="Raisen, Madhya Pradesh",
         problem="A conventional tractor-mounted cultivator could not reach proper tilling depth in the hard soil on his farm.",
         solution="He developed a combined tillage implement with a modified plough and PTO-powered auger that performs both primary and secondary tilling operations in a single pass."),
    dict(id="prafulla_weft", name="Prafulla Kumar Meher", place="Bargarh, Odisha",
         problem="Winding cotton thread by hand for weaving traditional Sambalpuri sarees was slow, and skilled labour for the job was becoming harder to find.",
         solution="He built a self-propelled weft-winding machine that automatically winds cotton thread bundles, sharply cutting the time needed compared to manual winding."),
    dict(id="ramprasad_weft", name="Ram Prasad Meher", place="Bargarh, Odisha",
         problem="The same slow, hand-powered thread-winding process used in Sambalpuri saree weaving was limiting how much a weaver's workshop could produce.",
         solution="Working independently, he built his own version of an automatic weft-winding machine that winds cotton bundles far faster than manual winding."),
    dict(id="suresh_arecanut", name="Suresh PV", place="Malappuram, Kerala",
         problem="Arecanut trees must be climbed several times a year for harvesting, a task only skilled, at-risk climbers could safely perform.",
         solution="He built a remote-controlled, petrol-engine-powered climbing machine that grips the tree trunk and climbs, harvests, and descends automatically."),
    dict(id="jitabhai_bean", name="Jitabhai Kodarbhai Patel", place="Sabarkantha, Gujarat",
         problem="Farmers growing hyacinth bean (val) lacked an early-maturing, high-yielding variety with good taste for the local Undhiyu dish market.",
         solution="Through eight years of selecting superior plants found mixed into collected fodder, he developed the JK-1 hyacinth bean variety, prized for its high yield and taste."),
    dict(id="laxmanbhai_bloat", name="Desai Laxmanbhai Devkaranbhai", place="Aravalli, Gujarat",
         problem="Livestock affected by bloat suffer painful abdominal swelling and reduced organ motility, and farmers had limited treatment options.",
         solution="Drawing on knowledge passed down from his father, he developed a herbal liquid preparation that, fed once daily, relieves bloat symptoms in affected animals."),
    dict(id="keisham_cauliflower", name="Keisham Thoibi Devi", place="Bishnupur, Manipur",
         problem="Farmers in her area lacked a traditional cauliflower cultivar with reliable early maturity and disease resistance for their fields.",
         solution="Over years of selecting the best curds from a traditional variety she received from her in-laws, she stabilised and distributed an early-maturing, pest-tolerant cauliflower now grown widely in her district."),
    dict(id="dattatraya_grapes", name="Dattatraya Nanasaheb Kale", place="Solapur, Maharashtra",
         problem="Grape growers in his region needed black grape varieties with better sweetness, size, and export-quality shelf life than the standard Sharad Seedless variety.",
         solution="By selecting and grafting naturally occurring mutations he spotted in his vineyard, he developed two new varieties, Sarita Seedless and Nanasaheb Purple Seedless, now grown across tens of thousands of hectares."),
    dict(id="roshan_pestcontrol", name="Roshan Lal", place="Kullu, Himachal Pradesh",
         problem="Insect pests were damaging wheat, barley, and vegetable crops for farmers in his district, cutting into their harvests.",
         solution="He developed a decoction from local plant bark, leaves, and fruit that controls sucking and borer insects in crops when used at low doses."),
    dict(id="jyotsna_pestcontrol", name="Jyotsna Mayee Patra", place="Keonjhar, Odisha",
         problem="Insect infestations were damaging rose plants and vegetable and rice crops in her home garden and fields.",
         solution="After noticing a local climber plant's decaying fruit kept insects away from nearby roses, she developed a preparation from the same plant that controls insect pests in vegetables and rice when applied to the soil."),
    dict(id="dayaram_tamarind", name="Dayaram Vishram Chouhan", place="Jeypore, Odisha",
         problem="A local non-profit needed a way to separate tamarind seeds from the pulp without relying on slow manual processing.",
         solution="He built a roller-based tamarind de-seeding machine, operable manually or by motor, and has sold over 200 units to organizations processing tamarind."),
    dict(id="kacharu_bloat", name="Kacharu Katara", place="Dungarpur, Rajasthan",
         problem="Livestock in his community regularly suffered from bloat, a painful digestive condition, with few treatment options available to farmers.",
         solution="A traditional healer, he developed a herbal leaf-powder remedy that, administered orally once a day, relieves bloat symptoms in affected animals."),
    dict(id="vijayram_bloat", name="Vijay Ram", place="Nainital, Uttarakhand",
         problem="Farmers in his hill district had no quick remedy for livestock suffering from bloat.",
         solution="Using a herbal treatment learned from family elders, he developed a dried-herb powder dose that relieves bloat symptoms in animals within a few hours."),
    dict(id="imna_optable", name="Imna Meren", place="Dimapur, Nagaland",
         problem="The local veterinary centre had no proper table for restraining and treating sick animals, forcing multiple people to hold a distressed animal down, and the centre itself was often far away over poor roads.",
         solution="He built a hydraulically adjustable wooden operation table for small animals with straps, an IV stand, and drainage, letting basic veterinary procedures be done safely closer to home."),
    dict(id="ravi_jacquard", name="P Ravi", place="Tiruvannamalai, Tamil Nadu",
         problem="Weavers operating Jacquard handlooms had to manually lift a heavy Jacquard box hundreds of times a day using a leg and wooden lever, causing physical strain and lower productivity.",
         solution="He designed a motorised attachment that lifts the Jacquard box at the press of a pedal, and has sold thousands of units to handloom weavers."),
    dict(id="sekar_weaving", name="P. A. Sekar", place="Thiruvallur, Tamil Nadu",
         problem="Sizing threads before weaving was traditionally done by hand every 19 metres, a slow, labour-intensive step in saree production.",
         solution="He built a machine that sizes threads over much longer 200-metre runs, multiplying daily output roughly twelvefold for the weaving families who use it."),
    dict(id="durlov_teadryer", name="Durlov Gogoi", place="Dibrugarh, Assam",
         problem="Selling fresh green tea leaves to large factories earned small tea garden owners like him very little per bag, and existing tea dryers he tried weren't good enough to process his own leaves profitably.",
         solution="After years of trial and error, he built a reciprocating 'push-pull' tea dryer that dries tea leaves efficiently, letting small growers process and sell their own dried tea."),
    dict(id="suren_coolingbed", name="Suren Barua", place="Tinsukia, Assam",
         problem="People without access to air conditioning struggled to stay comfortable during hot summer nights, especially the sick and elderly who spend long periods in bed; small tea growers also needed an affordable, lower-maintenance way to dry green tea leaves.",
         solution="He built a steel-frame cooling bed with a built-in air duct and compressor that cools the sleeping surface, and separately developed a belt-driven tea dryer that reduces the tray-jamming and high maintenance of existing dryers."),
    dict(id="tirupathi_pole", name="Nannem Tirupathi Rao", place="Prakasham, Andhra Pradesh",
         problem="As an electrician who installs power lines, he found climbing cement electric poles especially difficult and risky compared to wooden ones.",
         solution="He designed a bent steel-rod climbing frame fitted with sandals that lets a worker climb any type of electric pole safely, tested to hold up to 150 kilograms."),
    dict(id="arjunbhai_crematorium", name="Arjunbhai M. Paghdar", place="Junagadh, Gujarat",
         problem="Traditional Hindu cremation burns around 400 kilograms of wood per body, consuming millions of tonnes of wood nationally each year.",
         solution="He designed a biomass-gasification cremation structure lined with heat-retaining refractory brick and cera-wool that cremates a body using only 70 to 80 kilograms of wood in under half the usual time."),
    dict(id="durgadevi_crutch", name="Durga Devi", place="Bareilly, Uttar Pradesh",
         problem="As a physically challenged homemaker, she found that standard crutches slipped on wet or smooth floors and tended to slide out from under her arm.",
         solution="She modified ordinary crutches with a shoulder belt, a shock absorber, and a vacuum-cup base that grips smooth or wet flooring, preventing slips."),
    dict(id="babasaheb_onion", name="Babasaheb Nanabhau Pisore", place="Beed, Maharashtra",
         problem="Onion farmers in his area needed a variety with better shelf life and resistance to splitting and blotch disease than what was available.",
         solution="Through five years of mass selection from an existing variety, he developed Sona-40, an early-maturing onion with good shelf life and disease resistance, now distributed to farmers in a dozen states."),
    dict(id="roymathew_nutmeg", name="Roy Mathew", place="Idukki, Kerala",
         problem="When rubber prices collapsed, he shifted to nutmeg farming but lacked a variety producing consistently high yields of high-quality mace.",
         solution="By propagating and selecting from unusual local nutmeg varieties over years, he developed Souwriyamakkal, a nutmeg variety with unusually aromatic, oil-rich mace."),
    dict(id="helendro_chilli", name="Leimapokpam Helendro Singh", place="Imphal West, Manipur",
         problem="Farmers growing traditional Manipuri chilli varieties had to choose between a tastier, low-yielding variety and a higher-yielding one with smaller fruit.",
         solution="By selecting unusually large fruit from a high-yielding traditional variety over ten years, he developed the Helen Morok chilli, which combines good yield, big fruit, and disease tolerance."),
    dict(id="vithal_grapes", name="Vithal Nivruti Thorat", place="Pune, Maharashtra",
         problem="Grape growers wanted a black grape variety with bigger bunches, higher sugar content, and better shelf life than the standard Sharad Seedless.",
         solution="By selecting and grafting distinct vines he spotted in his own orchard, he developed the Nath Jumbo Seedless grape, prized for its bold berries, sweetness, and export-quality shelf life."),
    dict(id="kishansingh_pestcontrol", name="Kishan Singh", place="Kullu, Himachal Pradesh",
         problem="Insects and fungus were damaging cabbage, cauliflower, and other vegetable crops in his district.",
         solution="Using knowledge from his family elders, he developed a herbal-and-cow-urine preparation that controls insects and fungal infection in vegetable crops."),
    dict(id="divya_pestcontrol", name="Divya Sharma", place="Kullu, Himachal Pradesh",
         problem="Insects and termites were damaging fruit orchards and vegetable crops on her farm.",
         solution="She developed a fermented preparation from the fruit of a local tree that protects apples, pears, plums, and vegetables from insects and termites."),
    dict(id="laxmiben_bloat", name="Laxmiben Pratapji Thakarda", place="Banaskantha, Gujarat",
         problem="Livestock in her village regularly suffered from bloat, and farmers had few treatment options on hand.",
         solution="A recognised village veterinary healer, she developed a herbal leaf remedy that relieves bloat symptoms in animals within hours of dosing."),
    dict(id="shamalbhai_mastitis", name="Shamalbhai Kanabhai Gamar", place="Sabarkantha, Gujarat",
         problem="Dairy animals with mastitis suffered painful, inflamed udders and bloody milk, with limited treatment options for farmers in his village.",
         solution="Using a remedy passed down from his father, he developed a herbal treatment administered directly into the udder that reduces inflammation and curtails bacterial infection over about ten days."),
    dict(id="jhabarmal_harvester", name="Jhabarmal Khokhar", place="Sikar, Rajasthan",
         problem="Existing tractor-mounted cutters for harvesting low-height crops like green gram, cumin, and soybean worked poorly, leaving farmers without an effective mechanized option.",
         solution="He built a tractor-mounted multi-crop harvester that can also operate while the tractor moves in reverse, letting it harvest crops sown in any planting pattern."),
    dict(id="lakhanlal_sunflower", name="Lakhanlal Patel", place="Raigarh, Chhattisgarh",
         problem="Farmers growing sunflower at scale in his village struggled to extract seeds without expensive machinery, cutting into their returns.",
         solution="He built a low-cost, motor-powered sunflower seed extractor using rubber-belt beaters that deseeds sunflowers with over 95 percent efficiency."),
    dict(id="saruj_reeling", name="Saruj Chetia", place="Sivasagar, Assam",
         problem="Traditional Muga silk reeling was slow and labour-intensive, and commercial reeling machines were too costly for him to afford.",
         solution="He built a low-cost reeling machine, refined from an initial sewing-machine-parts prototype, that reels both warp and weft silk yarn simultaneously."),
    dict(id="jagtap_sandalwood", name="Subhash Vasantrao Jagtap", place="Jalgaon, Maharashtra",
         problem="Temples that need a continuous supply of pure sandalwood paste for religious rituals relied on the slow, laborious process of grinding it by hand.",
         solution="A veteran fabricator, he built a motorised rotary grinder that produces smooth sandalwood paste from small logs, saving temples the manual grinding work."),
    dict(id="sapan_swing", name="Sapan Kumar Mandal", place="Nadia, West Bengal",
         problem="He wanted to build his newborn daughter a swing that could adapt safely to her needs as she grew, from infant to toddler to young child.",
         solution="He designed a rail-guided sliding swing that can be reconfigured over time, from an infant's bed to a rod-protected toddler seat to a backed seat for an older child."),
    dict(id="mohan_cowdung", name="Mohan Muktaji Lamb", place="Beed, Maharashtra",
         problem="Dairy farm labourers were reluctant to collect cow dung by hand, and farms often faced a shortage of workers willing to do the job.",
         solution="He built a battery-powered machine with picking and conveying trays that collects cow dung without any physical contact required from the operator."),
    dict(id="towseef_bukhari", name="Towseef Ali Malik", place="Kishtiwar, Jammu & Kashmir",
         problem="The traditional Kashmiri bukhari room heater burned through 3 to 4 kilograms of wood or coal an hour and retained heat for only about half an hour after the fire went out.",
         solution="He modified the bukhari with gypsum and baked-clay logs that retain heat far longer, cutting fuel use to under a kilogram an hour while holding warmth for three to four hours."),
    dict(id="lanu_loadcontroller", name="Lanu L Jamir", place="Dimapur, Nagaland",
         problem="Small hydropower systems used in his region produce electricity that needs a stabilising controller to safely run household appliances, but imported controllers were too costly for local users.",
         solution="He designed an affordable electronic load controller that regulates voltage, speed, and frequency from small hydropower generators, sized for both 5kW and 100kW systems."),
    dict(id="sourav_lac", name="Sourav Dey", place="Singhbhum, Jharkhand",
         problem="Scraping lac resin from tree branches by hand with a knife or by beating sticks was slow, limited a worker to just a few kilograms a day, and left the scraped lac full of dirt and impurities.",
         solution="Inspired by a sugarcane juice extractor, he designed a manually operated roller-type lac scraper that removes lac from branches without breaking them, leaving it cleaner and easier to process."),
    dict(id="zufa_namda", name="Zufa Iqbal", place="Srinagar, Jammu & Kashmir",
         problem="Making a traditional Kashmiri namda woollen felt rug by hand, rolling and compressing wool for about an hour per piece, took a long time to complete even one.",
         solution="As a school student, she designed a motorised rolling machine with adjustable rollers that presses and fuses wool fibres into a finished namda in under ten minutes."),
    dict(id="evthomas_nutmeg", name="E. V. Thomas", place="Idukki, Kerala",
         problem="After losing his nutmeg orchard to disease, he needed a new, higher-quality nutmeg variety to rebuild his farm's income.",
         solution="By propagating and selecting plants with an unusual lateral branching pattern that made harvesting easier, he developed the Edavarembil Gold nutmeg variety, prized for its high oil content and consistent yield."),
]

# ---------------------------------------------------------------------------
# Paragraph paraphrase templates. Each takes (name, place, problem, solution)
# and produces one original-wording paragraph. Not all templates fit every
# case grammatically, so we keep them generic.
# ---------------------------------------------------------------------------
def strip_final_period(s):
    return s[:-1] if s.endswith(".") else s

def t1(c):
    return (f"In {c['place']}, {c['problem'][0].lower() + c['problem'][1:]} "
            f"{c['name']} responded with a practical fix: {c['solution'][0].lower() + c['solution'][1:]}")

def t2(c):
    return (f"{c['name']}, from {c['place']}, noticed that {c['problem'][0].lower() + c['problem'][1:]} "
            f"To solve this, {c['solution'][0].lower() + c['solution'][1:]}")

def t3(c):
    return (f"{c['problem']} That was the situation {c['name']} set out to change in {c['place']}. "
            f"{c['solution']}")

def t4(c):
    return (f"Faced with a familiar rural problem — {strip_final_period(c['problem'][0].lower() + c['problem'][1:])} — "
            f"{c['name']} came up with a grassroots solution. {c['solution']}")

def t5(c):
    return (f"This is the story of {c['name']}, a grassroots innovator from {c['place']}. "
            f"{c['problem']} {c['solution']}")

TEMPLATES = [t1, t2, t3, t4, t5]

# ---------------------------------------------------------------------------
# Build rows: 4-5 paragraph variants per case, same problem/solution label.
# ---------------------------------------------------------------------------
rows = []
for c in CASES:
    n_variants = 4 if len(c["problem"]) + len(c["solution"]) > 220 else 5
    chosen = random.sample(TEMPLATES, k=n_variants)
    for i, tmpl in enumerate(chosen):
        text = tmpl(c)
        rows.append({
            "case_id": c["id"],
            "row_id": f"{c['id']}__{i}",
            "text": text,
            "label": {"problem": c["problem"], "solution": c["solution"]},
        })

print(f"Total cases: {len(CASES)}")
print(f"Total rows:  {len(rows)}")

# ---------------------------------------------------------------------------
# Split by CASE (not by row) 80/10/10 to avoid leakage.
# ---------------------------------------------------------------------------
case_ids = list({c["id"] for c in CASES})
random.shuffle(case_ids)

n = len(case_ids)
n_train = int(round(n * 0.8))
n_val = int(round(n * 0.1))
train_ids = set(case_ids[:n_train])
val_ids = set(case_ids[n_train:n_train + n_val])
test_ids = set(case_ids[n_train + n_val:])

print(f"Cases -> train: {len(train_ids)}, val: {len(val_ids)}, test: {len(test_ids)}")

splits = {"train": [], "val": [], "test": []}
for r in rows:
    if r["case_id"] in train_ids:
        splits["train"].append(r)
    elif r["case_id"] in val_ids:
        splits["val"].append(r)
    else:
        splits["test"].append(r)

for name, split_rows in splits.items():
    random.shuffle(split_rows)
    path = f"data/{name}.jsonl"
    with open(path, "w") as f:
        for r in split_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name}: {len(split_rows)} rows -> {path}")

# Sanity check: no case_id appears in more than one split
assert not (train_ids & val_ids)
assert not (train_ids & test_ids)
assert not (val_ids & test_ids)
print("No case-level leakage between splits: OK")
