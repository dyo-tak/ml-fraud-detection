const categories = [
  "misc_net",
  "grocery_pos",
  "entertainment",
  "gas_transport",
  "misc_pos",
  "grocery_net",
  "shopping_net",
  "shopping_pos",
  "food_dining",
  "personal_care",
  "health_fitness",
  "travel",
  "kids_pets",
  "home",
];

const jobs = [
  "Psychologist, counselling",
  "Special educational needs teacher",
  "Nature conservation officer",
  "Patent attorney",
  "Dance movement psychotherapist",
  "Transport planner",
  "Arboriculturist",
  "Designer, multimedia",
  "Public affairs consultant",
  "Pathologist",
  "IT trainer",
  "Systems developer",
  "Engineer, land",
  "Systems analyst",
  "Naval architect",
  "Radiographer, diagnostic",
  "Programme researcher, broadcasting/film/video",
  "Energy engineer",
  "Event organiser",
  "Operational researcher",
  "Market researcher",
  "Probation officer",
  "Leisure centre manager",
  "Corporate investment banker",
  "Therapist, occupational",
  "Call centre manager",
  "Police officer",
  "Education officer, museum",
  "Physiotherapist",
  "Network engineer",
  "Forensic psychologist",
  "Geochemist",
  "Armed forces training and education officer",
  "Designer, furniture",
  "Optician, dispensing",
  "Psychologist, forensic",
  "Librarian, public",
  "Fine artist",
  "Scientist, research (maths)",
  "Research officer, trade union",
  "Tourism officer",
  "Human resources officer",
  "Surveyor, minerals",
  "Applications developer",
  "Video editor",
  "Curator",
  "Research officer, political party",
  "Engineer, mining",
  "Education officer, community",
  "Physicist, medical",
  "Amenity horticulturist",
  "Electrical engineer",
  "Television camera operator",
  "Higher education careers adviser",
  "Ambulance person",
  "Dealer",
  "Paediatric nurse",
  "Trading standards officer",
  "Engineer, technical sales",
  "Designer, jewellery",
  "Clinical biochemist",
  "Engineer, electronics",
  "Water engineer",
  "Science writer",
  "Film/video editor",
  "Solicitor, Scotland",
  "Product/process development scientist",
  "Tree surgeon",
  "Careers information officer",
  "Geologist, engineering",
  "Counsellor",
  "Freight forwarder",
  "Senior tax professional/tax inspector",
  "Engineer, broadcasting (operations)",
  "English as a second language teacher",
  "Economist",
  "Child psychotherapist",
  "Claims inspector/assessor",
  "Tourist information centre manager",
  "Exhibitions officer, museum/gallery",
  "Location manager",
  "Engineer, biomedical",
  "Research scientist (physical sciences)",
  "Purchasing manager",
  "Editor, magazine features",
  "Operations geologist",
  "Interpreter",
  "Engineering geologist",
  "Agricultural consultant",
  "Paramedic",
  "Financial adviser",
  "Administrator, education",
  "Educational psychologist",
  "Financial trader",
  "Audiological scientist",
  "Scientist, audiological",
  "Administrator, charities/voluntary organisations",
  "Health service manager",
  "Retail merchandiser",
  "Telecommunications researcher",
  "Exercise physiologist",
  "Accounting technician",
  "Product designer",
  "Waste management officer",
  "Mining engineer",
  "Surgeon",
  "Therapist, horticultural",
  "Environmental consultant",
  "Broadcast presenter",
  "Producer, radio",
  "Engineer, communications",
  "Historic buildings inspector/conservation officer",
  "Teacher, English as a foreign language",
  "Materials engineer",
  "Health visitor",
  "Medical secretary",
  "Theatre director",
  "Technical brewer",
  "Land/geomatics surveyor",
  "Engineer, structural",
  "Diagnostic radiographer",
  "Television production assistant",
  "Medical sales representative",
  "Building control surveyor",
  "Therapist, sports",
  "Structural engineer",
  "Commercial/residential surveyor",
  "Database administrator",
  "Exhibition designer",
  "Training and development officer",
  "Mechanical engineer",
  "Medical physicist",
  "Administrator",
  "Mudlogger",
  "Fisheries officer",
  "Conservator, museum/gallery",
  "Programmer, multimedia",
  "Cytogeneticist",
  "Multimedia programmer",
  "Counselling psychologist",
  "Chiropodist",
  "Teacher, early years/pre",
  "Cartographer",
  "Pensions consultant",
  "Primary school teacher",
  "Electronics engineer",
  "Museum/gallery exhibitions officer",
  "Air broker",
  "Chemical engineer",
  "Advertising account executive",
  "Advertising account planner",
  "Chartered legal executive (England and Wales)",
  "Psychiatric nurse",
  "Secondary school teacher",
  "Librarian, academic",
  "Embryologist, clinical",
  "Immunologist",
  "Television floor manager",
  "Contractor",
  "Health physicist",
  "Copy",
  "Bookseller",
  "Land",
  "Chartered loss adjuster",
  "Occupational psychologist",
  "Facilities manager",
  "Further education lecturer",
  "Archivist",
  "Investment analyst",
  "Engineer, building services",
  "Psychologist, sport and exercise",
  "Journalist, newspaper",
  "Doctor, hospital",
  "Phytotherapist",
  "Pharmacologist",
  "Horticultural therapist",
  "Hydrologist",
  "Community arts worker",
  "Public house manager",
  "Architect",
  "Lexicographer",
  "Psychotherapist, child",
  "Teacher, secondary school",
  "Toxicologist",
  "Commercial horticulturist",
  "Podiatrist",
  "Building surveyor",
  "Architectural technologist",
  "Editor, film/video",
  "Social researcher",
  "Wellsite geologist",
  "Minerals surveyor",
  "Designer, ceramics/pottery",
  "Mental health nurse",
  "Volunteer coordinator",
  "Chief Technology Officer",
  "Camera operator",
  "Copywriter, advertising",
  "Surveyor, mining",
  "Product manager",
  "Nurse, children's",
  "Pension scheme manager",
  "Archaeologist",
  "Sub",
  "Designer, interior/spatial",
  "Futures trader",
  "Chief Financial Officer",
  "Museum education officer",
  "Quantity surveyor",
  "Physiological scientist",
  "Loss adjuster, chartered",
  "Pilot, airline",
  "Production assistant, radio",
  "Immigration officer",
  "Retail banker",
  "Health and safety adviser",
  "Teacher, special educational needs",
  "Jewellery designer",
  "Community pharmacist",
  "Control and instrumentation engineer",
  "Make",
  "Early years teacher",
  "Sales professional, IT",
  "Scientist, marine",
  "Intelligence analyst",
  "Clinical research associate",
  "Administrator, local government",
  "Barrister",
  "Engineer, control and instrumentation",
  "Clothing/textile technologist",
  "Development worker, community",
  "Art therapist",
  "Sales executive",
  "Armed forces logistics/support/administrative officer",
  "Optometrist",
  "Insurance underwriter",
  "Charity officer",
  "Civil Service fast streamer",
  "Retail buyer",
  "Magazine features editor",
  "Equities trader",
  "Trade mark attorney",
  "Research scientist (life sciences)",
  "Psychotherapist",
  "Pharmacist, community",
  "Risk analyst",
  "Engineer, maintenance",
  "Logistics and distribution manager",
  "Water quality scientist",
  "Lecturer, further education",
  "Production assistant, television",
  "Tour manager",
  "Music therapist",
  "Surveyor, land/geomatics",
  "Engineer, production",
  "Acupuncturist",
  "Hospital doctor",
  "Teacher, primary school",
  "Accountant, chartered public finance",
  "Illustrator",
  "Scientist, physiological",
  "Buyer, industrial",
  "Scientist, research (physical sciences)",
  "Radio producer",
  "Manufacturing engineer",
  "Animal technologist",
  "Production engineer",
  "Biochemist, clinical",
  "Engineer, manufacturing",
  "Comptroller",
  "General practice doctor",
  "Designer, industrial/product",
  "Prison officer",
  "Merchandiser, retail",
  "Engineer, drilling",
  "Engineer, petroleum",
  "Cabin crew",
  "Commissioning editor",
  "Accountant, chartered certified",
  "Local government officer",
  "Professor Emeritus",
  "Press sub",
  "Chartered public finance accountant",
  "Writer",
  "Chief Executive Officer",
  "Occupational hygienist",
  "Doctor, general practice",
  "Community education officer",
  "Landscape architect",
  "Occupational therapist",
  "Special effects artist",
  "Civil engineer, contracting",
  "Barrister's clerk",
  "Travel agency manager",
  "Associate Professor",
  "Neurosurgeon",
  "Plant breeder/geneticist",
  "Radio broadcast assistant",
  "Field seismologist",
  "Industrial/product designer",
  "Metallurgist",
  "Politician's assistant",
  "Insurance claims handler",
  "Theme park manager",
  "Gaffer",
  "Chief Strategy Officer",
  "Heritage manager",
  "Ceramics designer",
  "Animator",
  "Oceanographer",
  "Colour technologist",
  "Engineer, agricultural",
  "Therapist, drama",
  "Orthoptist",
  "Learning mentor",
  "Arts development officer",
  "Biomedical engineer",
  "Race relations officer",
  "Therapist, music",
  "Retail manager",
  "Furniture designer",
  "Building services engineer",
  "Maintenance engineer",
  "Aid worker",
  "Editor, commissioning",
  "Private music teacher",
  "Scientist, biomedical",
  "Public relations account executive",
  "Dispensing optician",
  "Advice worker",
  "Hydrographic surveyor",
  "Geoscientist",
  "Environmental health practitioner",
  "Learning disability nurse",
  "Chief Operating Officer",
  "Scientific laboratory technician",
  "Records manager",
  "Barista",
  "Marketing executive",
  "Tax inspector",
  "Musician",
  "Therapist, art",
  "Engineer, automotive",
  "Clinical psychologist",
  "Warden/ranger",
  "Surveyor, rural practice",
  "Sport and exercise psychologist",
  "Education administrator",
  "Chief of Staff",
  "Nurse, mental health",
  "Music tutor",
  "Planning and development surveyor",
  "Teaching laboratory technician",
  "Chief Marketing Officer",
  "Theatre manager",
  "Quarry manager",
  "Interior and spatial designer",
  "Lecturer, higher education",
  "Regulatory affairs officer",
  "Secretary/administrator",
  "Chemist, analytical",
  "Designer, exhibition/display",
  "Pharmacist, hospital",
  "Site engineer",
  "Equality and diversity officer",
  "Public librarian",
  "Town planner",
  "Chartered accountant",
  "Programmer, applications",
  "Manufacturing systems engineer",
  "Web designer",
  "Community development worker",
  "Animal nutritionist",
  "Petroleum engineer",
  "Information systems manager",
  "Press photographer",
  "Insurance risk surveyor",
  "Soil scientist",
  "Buyer, retail",
  "Public relations officer",
  "Health promotion specialist",
  "Psychiatrist",
  "Visual merchandiser",
  "Rural practice surveyor",
  "Hotel manager",
  "Communications engineer",
  "Insurance broker",
  "Radiographer, therapeutic",
  "Set designer",
  "Tax adviser",
  "Drilling engineer",
  "Fitness centre manager",
  "Farm manager",
  "Management consultant",
  "Energy manager",
  "Museum/gallery conservator",
  "Herbalist",
  "Osteopath",
  "Statistician",
  "Hospital pharmacist",
  "Estate manager/land agent",
  "Sports development officer",
  "Investment banker, corporate",
  "Biomedical scientist",
  "Television/film/video producer",
  "Nutritional therapist",
  "Company secretary",
  "Production manager",
  "Magazine journalist",
  "Media buyer",
  "Data scientist",
  "Engineer, civil (contracting)",
  "Herpetologist",
  "Garment/textile technologist",
  "Scientist, research (medical)",
  "Civil Service administrator",
  "Airline pilot",
  "Textile designer",
  "Environmental manager",
  "Furniture conservator/restorer",
  "Horticultural consultant",
  "Firefighter",
  "Geophysicist/field seismologist",
  "Psychologist, clinical",
  "Development worker, international aid",
  "Sports administrator",
  "IT consultant",
  "Presenter, broadcasting",
  "Outdoor activities/education manager",
  "Field trials officer",
  "Social research officer, government",
  "English as a foreign language teacher",
  "Restaurant manager, fast food",
  "Hydrogeologist",
  "Research scientist (medical)",
  "Designer, television/film set",
  "Geneticist, molecular",
  "Designer, textile",
  "Licensed conveyancer",
  "Emergency planning/management officer",
  "Geologist, wellsite",
  "Air cabin crew",
  "Seismic interpreter",
  "Surveyor, hydrographic",
  "Charity fundraiser",
  "Stage manager",
  "Aeronautical engineer",
  "Glass blower/designer",
  "Ecologist",
  "Horticulturist, commercial",
  "Research scientist (maths)",
  "Engineer, aeronautical",
  "Conservation officer, historic buildings",
  "Art gallery manager",
  "Advertising copywriter",
  "Engineer, civil (consulting)",
  "Oncologist",
  "Engineer, materials",
  "Scientist, clinical (histocompatibility and immunogenetics)",
  "Investment banker, operational",
  "Medical technical officer",
  "Academic librarian",
  "Artist",
  "Clinical cytogeneticist",
  "TEFL teacher",
  "Administrator, arts",
  "Teacher, adult education",
  "Catering manager",
  "Environmental education officer",
  "Conservator, furniture",
  "Analytical chemist",
  "Broadcast engineer",
  "Media planner",
  "Lawyer",
  "Producer, television/film/video",
  "Armed forces technical officer",
  "Engineer, site",
  "Contracting civil engineer",
  "Veterinary surgeon",
  "Sales promotion account executive",
  "Broadcast journalist",
  "Dancer",
  "Forest/woodland manager",
  "Personnel officer",
  "Industrial buyer",
  "Accountant, chartered",
  "Air traffic controller",
  "Careers adviser",
  "Information officer",
  "Ship broker",
  "Legal secretary",
  "Homeopath",
  "Solicitor",
  "Warehouse manager",
];

const cities = ['Moravian Falls', 'Orient', 'Malad City', 'Boulder', 'Doe Hill',
'Dublin', 'Holcomb', 'Edinburg', 'Manor', 'Clarksville',
'Clarinda', 'Shenandoah Junction', 'Saint Petersburg', 'Grenada',
'High Rolls Mountain Park', 'Harrington Park', 'Lahoma',
'Carlisle', 'Harborcreek', 'Elizabeth', 'Methuen', 'Moulton',
'Plainfield', 'May', 'Waukesha', 'Bailey', 'Romulus', 'Freedom',
'Honokaa', 'Valentine', 'Westfir', 'Tiptonville', 'Republic',
'Baton Rouge', 'Washington', 'Big Creek', 'Bellmore', 'Florence',
'Allentown', 'Moriches', 'Esbon', 'Chatham', 'Thompson',
'North Prairie', 'Laredo', 'Grant', 'Conway', 'New Goshen',
'Sunflower', 'Enola', 'Roosevelt', 'Pointe Aux Pins', 'Dallas',
'Jay', 'North Tonawanda', 'Athena', 'Chester', 'Elkhart',
'Surrency', 'Arcadia', 'Gaithersburg', 'Bowdoin', 'Heart Butte',
'San Jose', 'Rumely', 'Cranks', 'Ravenna', 'Utica', 'Uledi',
'Naples', 'Thida', 'Parks', 'Central', 'Fort Washakie', 'Etlan',
'Brinson', 'Shrewsbury', 'Bigelow', 'North Washington', 'Holloway',
'Littleton', 'Hinesburg', 'Meadville', 'Elberta', 'Moab',
'Diamond', 'Bradley', 'Hopewell', 'Hawthorne', 'Kensington',
'Ruth', 'Avoca', 'Princeton', 'Sherman', 'Loxahatchee',
'Smackover', 'Cokeburg', 'Leetsdale', 'Manville',
'Westhampton Beach', 'New York City', 'Halstad', 'Allenhurst',
'June Lake', 'Sixes', 'Holstein', 'Brantley', 'Paxton',
'Westerville', 'Timberville', 'Dunlevy', 'Coyle', 'Elizabethtown',
'Garrattsville', 'Lomax', 'Egan', 'Brownville', 'Drakes Branch',
'Ballwin', 'Fields Landing', 'West Palm Beach', 'West Chazy',
'Lebanon', 'Jones', 'Burke', 'Webster City', 'Ozawkie',
'Blackville', 'Damascus', 'Sea Island', 'Rule', 'Jefferson',
'Huntsville', 'Hinckley', 'Acworth', 'Camden', 'Smiths Grove',
'Philadelphia', 'Murrayville', 'Melbourne', 'Palmyra', 'Jermyn',
'Belfast', 'West Sayville', 'Sturgis', 'Lohrville', 'Arlington',
'Saint Paul', 'Girard', 'Fairview', 'Farmington', 'Kilgore',
'Springfield', 'Lepanto', 'Karnack', 'Louisiana', 'Kansas City',
'Mesa', 'Pomona', 'Lonetree', 'Port Saint Lucie', 'Centerview',
'Colorado Springs', 'Easton', 'West Columbia', 'Maysville',
'Muskegon', 'Johns Island', 'Metairie', 'Blairsden-Graeagle',
'Hampton', 'Viola', 'Paulding', 'Sontag', 'Clutier', 'Reno',
'Barnstable', 'Cardwell', 'Kingsford Heights', 'Phoenix',
'Newhall', 'Norwalk', 'Mc Nabb', 'Tulsa', 'Barnard', 'San Antonio',
'Bridger', 'Tomales', 'Mallie', 'Oxford', 'Turner', 'Eureka',
'Marietta', 'Redford', 'Weeping Water', 'Waynesfield', 'Portland',
'Greenwood', 'Boonton', 'Murfreesboro', 'Warren', 'Meridian',
'Cherokee Village', 'Bristow', 'Ash Flat', 'Iliff', 'Deadwood',
'Burlington', 'Wales', 'Churubusco', 'Afton', 'Cochranton',
'Mound City', 'West Harrison', 'Pembroke Township', 'Joliet',
'Fulton', 'Saint Bonaventure', 'Keisterville', 'Greenview',
'Lakeport', 'Rock Tavern', 'Llano', 'Altair', 'Harrodsburg',
'Carlotta', 'Wittenberg', 'Des Moines', 'Auburn', 'Dadeville',
'Wilton', 'Ollie', 'Mifflin', 'Amsterdam', 'Manderson', 'Holliday',
'Dumont', 'Payson', 'Birmingham', 'Creola', 'Texarkana',
'Fullerton', 'Ruckersville', 'Blooming Grove', 'Comfort',
'Louisville', 'Atglen', 'North Loup', 'Grassflat', 'Mill Creek',
'Moores Hill', 'Browning', 'Smithfield', 'Cromona', 'Kent',
'Harwood', 'Delhi', 'Gadsden', 'Brooklin', 'Bay Minette',
'Higganum', 'Ashfield', 'Fiddletown', 'Andrews',
'Huntington Beach', 'Westport', 'Comfrey', 'Heidelberg', 'Powell',
'De Witt', 'Howes Cave', 'Oolitic', 'Belmond', 'Mooresville',
'Steuben', 'Armagh', 'Edisto Island', 'Marathon', 'Port Gibson',
'Brandon', 'Tallmansville', 'Glendale', 'El Paso', 'Lamberton',
'Manchester', 'Emporium', 'Alva', 'Roma', 'Prairie Hill',
'Blairstown', 'Bridgeport', 'Grand Junction', 'Tamaroa',
'Great Mills', 'Laguna Hills', 'West Decatur', 'Aledo', 'Pelham',
'Albuquerque', 'Scotland', 'Azusa', 'Lanark Village',
'Cuyahoga Falls', 'Manistique', 'Mountain Park', 'University',
'Breesport', 'Harper', 'Bagley', 'Fenelton', 'Gardiner',
'Greendale', 'Bay City', 'Minneapolis', 'Gregory', 'Tuscarora',
'Northport', 'Rock Springs', 'Altona', 'Paauilo', 'Ratcliff',
'Mount Morris', 'Pembroke', 'Pittsburgh', 'West Henrietta',
'Tickfaw', 'Rhame', 'Eugene', 'Seneca', 'Daly City', 'Logan',
'Coleman', 'Norman', 'Cape Coral', 'Summerfield', 'Daniels',
'Tampa', 'Gainesville', 'East Canaan', 'Dresden', 'Luzerne',
'Parkers Lake', 'Olmsted', 'Mendon', 'Fayetteville', 'Notrees',
'Pecos', 'Brunson', 'Belgrade', 'Kenner', 'Stillwater', 'Cadiz',
'Avera', 'Annapolis', 'Powell Butte', 'Rosewood', 'Lima', 'Thrall',
'Union', 'Riverview', 'Hannawa Falls', 'Cazenovia', 'North Judson',
'Fordoche', 'Bauxite', 'Cisco', 'Apison', 'Cass', 'Santa Monica',
'Amorita', 'Laramie', 'Hudson', 'Center Point', 'Lagrange',
'Dexter', 'Brainard', 'Winger', 'Thomas', 'Shelter Island',
'Tupper Lake', 'Humboldt', 'Winter', 'Corona', 'Lakeland',
'Vienna', 'Goreville', 'Yellowstone National Park', 'Shedd',
'Wilmington', 'Mc Cracken', 'Milner', 'Noonan', 'Falconer',
'Desdemona', 'Bethel Springs', 'Burns Flat', 'Glen Rock', 'Basye',
'Lowville', 'Du Pont', 'Burrton', 'Lubbock', 'Independence',
'Creedmoor', 'Wichita', 'Houston', 'Clay Center', 'Sebring',
'Whittemore', 'Ogdensburg', 'Spearsville', 'Oaks',
'Paradise Valley', 'Burbank', 'New Holstein', 'Schaefferstown',
'Washington Court House', 'Eldridge', 'San Diego', 'Marienville',
'Fort Myers', 'Kissee Mills', 'Bronx', 'Montandon', 'Espanola',
'Campbell', 'Indian Wells', 'Hewitt', 'Leo', 'Mulberry Grove',
'Mountain Center', 'Helm', 'Detroit', 'Moundsville', 'Falmouth',
'Zaleski', 'Lowell', 'Henderson', 'Corsica', 'Wilmette',
'Mayersville', 'Pearlington', 'Cleveland', 'Whigham', 'Kirby',
'Knoxville', 'Highland', 'Hazel', 'Alexandria', 'Port Ewen',
'Lamy', 'Mount Clemens', 'North Wilkesboro', 'Morrisdale', 'Cord',
'Sauk Rapids', 'Center Tuftonboro', 'Hatch', 'Margaretville',
'Goodrich', 'Ford', 'South Londonderry', 'Cross Plains',
'Lake Jackson', 'Orangeburg', 'Walnut Ridge', 'Tower Hill',
'Oakland', 'Montrose', 'Parker Dam', 'Rocky Mount', 'Coleharbor',
'Falls City', 'Moriarty', 'Southfield', 'Glade Spring', 'Slayden',
'O Brien', 'Bristol', 'Umatilla', 'Hooper', 'Saint Amant',
'Clayton', 'Cascade Locks', 'Saint James City', 'Vinton',
'Christine', 'Alder', 'Roseland', 'Tyaskin', 'Lake Oswego',
'Parker', 'Wauchula', 'Colton', 'Juliette', 'River',
'Beaver Falls', 'Elk Rapids', 'Lonsdale', 'West Hartford',
'Kirtland Afb', 'Shields', 'Riverton', 'Whaleyville', 'Sacramento',
'Marion', 'Adams', 'Zavalla', 'Cottekill', 'Shippingport',
'Montgomery', 'Sun City', 'Grover', 'Newport',
'South Richmond Hill', 'Parsonsfield', 'Garfield', 'Jelm',
'Meredith', 'Battle Creek', 'Woodville', 'Luray', 'Cowlesville',
'Haw River', 'Falls Church', 'Topeka', 'Haynes', 'Grimesland',
'Smith River', 'Moorhead', 'White Sulphur Springs', 'East Andover',
'Superior', 'Hurricane', 'Bowersville', 'Mount Saint Joseph',
'Hancock', 'Dubre', 'Collettsville', 'Rossville', 'Red River',
'Benton', 'Monmouth Beach', 'Bessemer', 'Greenbush', 'Reynolds',
'Nazareth', 'Hovland', 'Alpharetta', 'Canton', 'Catawba',
'Scarborough', 'Hopkins', 'Norman Park', 'Sachse', 'Arvada',
'Winthrop', 'De Soto', 'Sutherland', 'Grandview', 'Manquin',
'Owensville', 'Altonah', 'Titusville', 'Paris', 'Humble', 'Mounds',
'Lolita', 'Wetmore', 'Keller', 'Amanda', 'Scotia', 'Pueblo',
'Purmela', 'Chester Heights', 'Clarks Mills', 'Mount Hope',
'West Eaton', 'Beaverdam', 'Randolph', 'Greenwich', 'Ragland',
'Phenix City', 'Coffeeville', 'Stirling', 'Issaquah',
'Old Hickory', 'West Long Branch', 'Ronceverte', 'Sula',
'Belle Fourche', 'Vero Beach', 'Curlew', 'Tyler', 'Miamisburg',
'Providence', 'Saint Louis', 'Duncan', 'De Lancey', 'Lawn',
'Tryon', 'Milwaukee', 'New Ellenton', 'East Troy', 'Saxon',
'West Green', 'Atlantic', 'Mc Veytown', 'Ferney', 'Corriganville',
'Pikesville', 'Cecilton', 'Wheaton', 'Palmdale', 'New Memphis',
'Newton', 'Valdosta', 'Jordan Valley', 'Edmond', 'Hills',
'Waupaca', 'Darien', 'Milford', 'De Queen', 'Newberg', 'Plymouth',
'Columbia', 'Belmont', 'Quanah', 'Barneveld', 'Newark Valley',
'Downsville', 'American Fork', 'Clifton', 'Stayton', 'Bonfield',
'Napa', 'West Monroe', 'Dayton', 'Hawley', 'Kingsport', 'Moro',
'Livonia', 'Sheffield', 'Gretna', 'Haines City', 'Key West',
'Stanchfield', 'Rockwood', 'Hahira', 'Munith', 'Minnesota Lake',
'New Waverly', 'Sprague', 'Dieterich', 'Mansfield', 'Winfield',
'Rochester', 'Oran', 'Orr', 'Crownpoint', 'Putnam', 'San Angelo',
'Eagarville', 'Leonard', 'Claremont', 'Spencer', 'Nelson',
'Rockwell', 'Ringwood', 'Oakdale', 'Nokomis', 'Bethel', 'Cedar',
'Loami', 'Spring Church', 'Brashear', 'Tekoa', 'Kingsville',
'Galatia', 'Dell City', 'Emmons', 'Norwich', 'Maria Stein',
'Georgetown', 'Clearwater', 'Albany', 'Indianapolis', 'Red Cliff',
'Omaha', 'Brooklyn', 'Richland', 'Arnold', 'Odessa', 'Iselin',
'Washoe Valley', 'Denham Springs', 'Mobile', 'Vanderbilt',
'Varnell', 'Halma', 'Oriskany Falls', 'Ranier', 'Lawrence',
'Hurley', 'Grand Bay', 'Sardis', 'Woods Cross', 'Broomfield',
'Schaumburg', 'Saint Francis', 'Heiskell', 'Unionville',
'Nobleboro', 'South Hero', 'Carroll', 'Achille', 'Lithopolis',
'Manley', 'Graniteville', 'Plantersville', 'Jaffrey',
'Liberty Mills', 'Stephensport', 'Matawan', 'Grantham', 'Tomahawk',
'Lakeview', 'Lane', 'Pewee Valley', 'Bolton', 'Matthews',
'Jordanville', 'Syracuse', 'Ruidoso', 'Oak Hill', 'Cressona',
'Early', 'Bryant', 'Bynum', 'Palermo', 'Fairhope', 'Remer',
'Baroda', 'Howells', 'Beasley', 'Phil Campbell', 'Deane',
'North Haverhill', 'Mount Perry', 'Prairie Creek', 'Grand Ridge',
'Oklahoma City', 'Ashford', 'Huslia', 'Irvine', 'Ehrhardt',
'Prosperity', 'Dongola', 'Paint Rock', 'Dalton', 'Smock',
'Bolivar', 'Spring', 'Ridgeland', 'West Finley', 'Lorenzo',
'Clune', 'Port Costa', 'Oconto Falls', 'Preston', 'Bonita Springs',
'Hartford', 'Hedrick', 'Boyd', 'Big Indian', 'Cuthbert',
'Sterling City', 'Queenstown', 'Trenton', 'Kirk', 'Gibsonville',
'North Brookfield', 'Akron', 'Monitor', 'New Boston', 'Monetta',
'Mc Intosh', 'Watertown', 'Alton', 'Rock Glen', 'East Rochester',
'Armonk', 'Stoneham', 'Winslow', 'Williamsburg', 'Mc Clellandtown',
'Hedley', 'Skytop', 'Cross', 'Deltona', 'Irwinton',
'North Augusta', 'Collegeville', 'Knowlesville', 'Oakford',
'Aurora', 'Heislerville', 'Scotts Mills', 'Moscow', 'Williams',
'Premier', 'Port Charlotte', 'Wendel', 'Harmony',
'Springfield Gardens', 'North Las Vegas', 'Port Richey',
'Linthicum Heights', 'Veedersburg', 'Loving', 'Stittville',
'Kittery Point', 'Greenville', 'Los Angeles', 'Ironton',
'Vancouver', 'Jackson', 'Kings Bay', 'Thornville', 'Cassatt',
'Michigan', 'Spirit Lake', 'Pea Ridge', 'West Bethel',
'New Franken', 'Waukau', 'Crouse', 'Vacaville', 'Madisonville',
'La Grande', 'Greenport', 'West Frankfort', 'Chattanooga',
'Gaines', 'Phelps', 'Hubbell', 'Mount Vernon', 'Beacon', 'Clinton',
'Claypool', 'Walkertown', 'Kaktovik', 'Coulee Dam',
'Mountain City', 'Granbury', 'Ashland', 'Clarion', 'Downey',
'Morven', 'Medford', 'Melville', 'Ridge Spring', 'Mineral',
'Orange Park', 'Norfolk', 'Irvington', 'Oakton', 'Wappapello',
'Nanuet', 'Seattle', 'Pleasant Hill', 'Karns City', 'North East',
'Byesville', 'Roland', 'Marshall', 'Isanti', 'Winnsboro',
'Noblesville', 'Las Vegas', 'Bruce', 'Buellton', 'Grenola',
'Streator', 'Wartburg', 'East China', 'Brookfield', 'Angwin',
'Nicholson', 'Lockhart', 'Moss Point', 'Queen Anne', 'Freeport'];

export default { categories, jobs, cities };
