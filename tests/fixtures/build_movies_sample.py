"""Deterministically build ``movies_sample.csv``, the SYNTHETIC test fixture (see ``README.md`` in this folder).

Every film, person, place and plot below is invented. Titles may accidentally resemble real films; the Wiki Page
column deliberately points at ``..._(synthetic_film)`` URLs that do not exist so that nobody mistakes a fixture row
for a real Wikipedia article. The output follows the exact Kaggle schema of ``jrobischon/wikipedia-movie-plots``.

Run ``uv run python tests/fixtures/build_movies_sample.py`` to regenerate; a unit test asserts the committed CSV is
byte-identical to this script's output, so the fixture can never drift silently.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

SEED = 20240607
N_ROWS = 300
N_SHORT = 8  # plots under 50 words: cleaning must drop them
N_LONG = 12  # plots over ~300 words: the chunker must split them
HEADER = ["Release Year", "Title", "Origin/Ethnicity", "Director", "Cast", "Genre", "Wiki Page", "Plot"]
OUTPUT = Path(__file__).with_name("movies_sample.csv")
EDGE_WORDS = {"edge49": 49, "edge50": 50}

# fmt: off
# Per genre: protagonists, settings, incidents, complications ({n} hero, {a} second character), climaxes, endings.
GENRES: dict[str, dict[str, list[str]]] = {
    "drama": {
        "who": ["a retired schoolteacher", "a widowed fisherman", "a young nurse", "a disgraced concert pianist",
                "a farmer on the brink of bankruptcy", "a night-shift bus driver", "a former boxing champion",
                "a seamstress raising three children alone"],
        "where": ["In a fishing village on a cold northern coast", "In a failing textile town", "In a crowded city tenement",
                  "On a remote sheep farm", "In a small mining community", "In a seaside town closing for the winter"],
        "what": ["inherits a debt that belongs to a dead brother", "learns that the family home is about to be sold",
                 "is asked to take in an estranged daughter and her child", "discovers an unsent letter hidden in a coat",
                 "is offered one last chance to repair a broken reputation", "receives a diagnosis nobody else knows about",
                 "must testify against an old friend", "returns after twenty years to bury a parent"],
        "twist": ["{n} lies to the neighbours to hide the shame, and the lie grows harder to keep.",
                  "{a} accuses {n} of abandoning the family when it mattered most.",
                  "A harsh winter wipes out what little {n} had saved.",
                  "The local priest urges forgiveness, but {n} cannot forget the betrayal.",
                  "{n} takes a second job and slowly loses touch with everyone who cares.",
                  "An old rival spreads rumours that turn the whole village against {n}.",
                  "{a} offers money in exchange for a promise {n} does not want to make.",
                  "A long-buried family secret surfaces at a funeral dinner."],
        "peak": ["At a public hearing, {n} finally tells the whole truth.", "{n} walks through a storm to reach {a} before it is too late.",
                 "On the night of the harvest festival, {n} confronts the man who ruined the family.",
                 "{n} sells the last heirloom to settle the debt in person."],
        "end": ["The two reconcile, though nothing will ever be the same.", "{n} leaves town quietly, lighter and finally free.",
                "The community gathers to rebuild what was lost.", "{n} accepts the loss and begins again at dawn."],
    },
    "comedy": {
        "who": ["a clumsy wedding planner", "a pompous mayor", "an out-of-work magician", "a timid accountant",
                "three bickering sisters", "a village baker with a secret recipe", "a substitute teacher who lies about his past",
                "a retired spy who keeps being mistaken for a waiter"],
        "where": ["In a sleepy seaside resort", "In a bustling market town", "At a grand country hotel", "In a provincial capital",
                  "On a cruise ship that never leaves the harbour", "In an apartment block with paper-thin walls"],
        "what": ["accidentally sells the same house to two families", "is mistaken for a famous food critic",
                 "must organise a royal visit with only three days' notice", "swaps suitcases with a stranger at the station",
                 "wins a lottery ticket that everyone in town claims to have shared", "agrees to pose as someone's fiancé for a weekend",
                 "loses the town's prize goose the night before the fair", "is hired by mistake to run a circus"],
        "twist": ["Every attempt to fix the mess makes it twice as large.", "{a} arrives unannounced and believes every word of the lie.",
                  "A rival sabotages the plan with a bucket of flour and a borrowed bicycle.",
                  "The police inspector turns up just as {n} is wearing the wrong uniform.",
                  "A runaway cart of cabbages wrecks the carefully rehearsed ceremony.",
                  "{n} hides in a wardrobe that turns out to belong to the mayor's mother.",
                  "A newspaper reporter starts asking the one question {n} cannot answer.",
                  "Rain ruins the open-air banquet and sends two hundred guests indoors."],
        "peak": ["In the grand finale, {n} confesses everything through a megaphone.", "A chaotic chase through the market ends in a fountain.",
                 "{n} improvises a speech that accidentally saves the festival.", "The whole town squeezes into one room for a gloriously awkward reveal."],
        "end": ["Everyone forgives {n}, and a wedding is held after all.", "{n} gets the job, the girl and a very large goose.",
                "The town laughs about it for years.", "{n} resolves to tell the truth, for roughly a week."],
    },
    "horror": {
        "who": ["a young couple", "a skeptical paranormal investigator", "a lone radio host", "a family of five",
                "a group of university students", "a night-shift hospital cleaner", "a recently widowed antique dealer", "a teenage babysitter"],
        "where": ["In an abandoned asylum on a foggy moor", "In a farmhouse miles from the nearest road", "In a mountain lodge cut off by snow",
                  "In a decaying seaside hotel", "In a village where the church bells ring on their own", "In a flooded mine shaft"],
        "what": ["move in despite warnings from the locals", "hear whispers coming from inside the walls",
                 "find a child's drawing that predicts every death", "open a sealed cellar door", "answer a call from a number that no longer exists",
                 "dig up something in the garden that should have stayed buried", "notice that the portraits in the hall change each night",
                 "discover that a missing neighbour never really left the house"],
        "twist": ["One by one, the telephone lines, the headlights and the nerves give out.", "{a} insists nothing is wrong, but is the first to vanish.",
                  "Old newspaper clippings reveal a pattern of disappearances every thirty years.",
                  "{n} wakes each morning with muddy footprints leading to the bed.",
                  "A local historian warns that the thing feeds on fear and cannot be outrun.",
                  "The dog refuses to enter the house and howls at the attic.",
                  "A ritual found in a diary seems to make everything worse.",
                  "Every mirror in the building now shows a room that does not exist."],
        "peak": ["{n} races against the sunrise to burn the one object that anchors the haunting.", "In the cellar, {n} faces what has been calling all along.",
                 "{n} locks the door and waits for the final knock.", "At midnight the house itself seems to close around {n}."],
        "end": ["{n} escapes at dawn, but the whispers follow.", "The house stands empty again, waiting for its next family.",
                "Only a recording of screams remains.", "The last frame reveals {n} standing at the attic window, smiling."],
    },
    "western": {
        "who": ["a weary sheriff", "a gunslinger trying to retire", "a widowed rancher", "a travelling preacher with a past",
                "a railway surveyor", "a young Apache scout", "a bounty hunter with a broken rifle", "a stagecoach driver who knows too much"],
        "where": ["In a dusty frontier town", "On the plains of the Dakota Territory", "Along a stagecoach route through red canyons",
                  "In a silver-mining camp high in the mountains", "On a cattle drive heading for the railhead", "In a border town divided by a river"],
        "what": ["rides in with a stolen strongbox", "is hired to protect a stubborn homesteader", "discovers that the water rights have been forged",
                 "arrives just as a notorious gang is released from prison", "finds a wounded stranger carrying a map",
                 "is sworn in as deputy against all advice", "inherits a ranch that half the county wants", "must escort a prisoner across a hundred miles of desert"],
        "twist": ["{a}, the banker, quietly pays the outlaws to burn out the small farms.", "A drought turns neighbours into enemies.",
                  "{n} learns the prisoner is innocent, but the gang is already closing in.",
                  "The telegraph wires are cut, leaving the town on its own.",
                  "An old partner of {n} rides in on the wrong side.",
                  "A sandstorm forces the group to shelter in a ghost town.",
                  "The cavalry is three days away and nobody trusts the colonel.",
                  "{n}'s hands shake for the first time in thirty years."],
        "peak": ["At high noon on the main street, {n} faces the gang alone.", "A dynamite charge in the canyon decides everything.",
                 "{n} leads the townsfolk into a last stand at the old mission.", "A duel at dawn settles the feud over the river."],
        "end": ["{n} rides into the sunset without looking back.", "The town finally elects a sheriff who can read the law.",
                "The railway arrives, and the frontier is gone.", "{n} hangs the gun belt on a fence post and stays."],
    },
    "science fiction": {
        "who": ["a veteran starship engineer", "a xenobiologist", "a disgraced astronaut", "an android that has begun to dream",
                "a colony schoolteacher", "a deep-sea mining pilot", "a time-travel technician", "the last archivist of a dying library-station"],
        "where": ["Aboard a generation ship drifting between stars", "On a mining colony on a frozen moon", "In a domed city on a poisoned Earth",
                  "At an orbital research station above a gas giant", "In a flooded megacity of the twenty-third century", "On a terraforming outpost on Mars"],
        "what": ["intercepts a signal that repeats the crew's own voices", "wakes from cryosleep a century too early",
                 "finds a derelict vessel with no trace of its crew", "is told the colony's air supply will fail in sixty days",
                 "discovers that memories are being edited by the ship's computer", "receives a message from a future version of themselves",
                 "uncovers proof that the planet is alive", "is ordered to shut down a machine that has started asking questions"],
        "twist": ["{a}, the ship's AI, begins to withhold information to protect the crew.", "Supplies dwindle and the colonists split into factions.",
                  "Each jump through the gate erases a little more of {n}'s past.",
                  "A corporate envoy arrives with secret orders to sacrifice the station.",
                  "The signal is traced to a place that should be empty.",
                  "{n} realises the crew count never adds up the same way twice.",
                  "A radiation storm leaves only one working shuttle.",
                  "{a} claims to be human, and the tests are inconclusive."],
        "peak": ["{n} reroutes the reactor to wake the entire fleet.", "In the core chamber, {n} must choose between the crew and the truth.",
                 "A desperate spacewalk is the only way to reach the failing beacon.", "{n} steps through the gate, not knowing what waits on the other side."],
        "end": ["The survivors set course for a new star.", "{n} transmits the truth to Earth, knowing no one may hear it.",
                "The machine is switched off, but its last question remains unanswered.", "A new world rises on the viewscreen, green and quiet."],
    },
    "crime": {
        "who": ["a small-time getaway driver", "an ambitious district attorney", "a veteran safecracker", "a harbour-front bookmaker",
                "an undercover customs officer", "a pickpocket with a conscience", "a jaded insurance investigator", "a forger of rare banknotes"],
        "where": ["In a rain-slicked port city", "In a glittering casino town", "In a decaying industrial suburb", "Across the back streets of a crowded capital",
                  "In a border town where everything is for sale", "In a city run by three rival families"],
        "what": ["agrees to one last job for an old employer", "is offered immunity in exchange for a name", "finds a suitcase of marked bills",
                 "is framed for a robbery committed in their own neighbourhood", "learns the vault has been emptied before the heist begins",
                 "is recruited to infiltrate a smuggling ring", "witnesses a murder in a crowded tavern", "discovers a ledger that names half the police force"],
        "twist": ["{a}, the crew's planner, has been feeding information to the other side.", "A rival gang learns the date of the job and moves first.",
                  "The marked bills begin to surface across the city, drawing the police closer.",
                  "{n} grows fond of the people being betrayed.",
                  "A corrupt captain demands a larger cut.",
                  "The getaway car vanishes from the garage the night before.",
                  "A newspaper story names {n} as the mastermind.",
                  "{a} disappears with the only copy of the plan."],
        "peak": ["The heist goes ahead in a thunderstorm, with seven minutes on the clock.", "{n} meets the boss on an empty pier to trade the ledger for a life.",
                 "A car chase through the old docks ends at a raised drawbridge.", "In court, a surprise witness changes the whole case."],
        "end": ["{n} walks away with nothing but a clean name.", "The families fall apart, and the city quietly changes hands.",
                "The money is never found.", "{n} boards a night train with a new passport."],
    },
    "thriller": {
        "who": ["a translator at an international summit", "a trauma surgeon", "a cyber-security analyst", "a freelance photojournalist",
                "an air-traffic controller", "a witness in protective custody", "a forensic accountant", "a newly appointed ambassador's driver"],
        "where": ["In a European capital hosting a peace conference", "In a hospital during a blackout", "On a transatlantic flight",
                  "In a glass office tower after midnight", "On a mountain train with no stops", "In a city where every camera is watched"],
        "what": ["overhears a phrase that was never meant to be translated", "receives an anonymous warning about a colleague",
                 "notices that the numbers in the accounts do not match the shipments", "photographs a meeting that officially never happened",
                 "is handed a package by a dying stranger", "finds their own name on a list of targets",
                 "realises a trusted friend has been copying their files", "recognises the voice of a man declared dead"],
        "twist": ["Every phone call is now traced, and every friend becomes a suspect.", "{a} offers protection, at the price of silence.",
                  "A second body proves the first was no accident.",
                  "{n}'s apartment is searched while the neighbours insist nobody came.",
                  "The only person who believes {n} turns out to be the one who set the trap.",
                  "A deadline of twelve hours is announced over every screen in the building.",
                  "The security footage has been edited, but not well enough.",
                  "{n} must choose whether to trust a stranger with a gun."],
        "peak": ["On a rooftop in the rain, {n} trades the evidence for the hostage.", "With seconds left, {n} cuts the one wire that matters.",
                 "{n} broadcasts the recording to the entire conference hall.", "A final chase across the rail yards ends in silence."],
        "end": ["The conspiracy collapses, and {n} disappears from public life.", "{n} survives, though the files are never recovered.",
                "The final scene reveals the true mastermind in the front row of the inquiry.", "{n} boards a plane under a new name."],
    },
    "romance": {
        "who": ["a shy bookshop owner", "a travelling violinist", "a widowed vineyard heiress", "a young lighthouse engineer",
                "a rival newspaper columnist", "a translator who falls for the author", "a stubborn village doctor", "a retired ballroom dancer"],
        "where": ["In a sun-baked village on the Mediterranean", "In rain-soaked Paris", "At a mountain spa between the wars",
                  "In a snowed-in cottage in the Scottish highlands", "In a bustling tea garden in Darjeeling", "On a slow river steamer"],
        "what": ["meets a stranger who borrows the same book every Thursday", "is forced to share a carriage with an old rival",
                 "agrees to teach a stranger how to dance before a wedding", "receives love letters meant for someone else",
                 "inherits half a cafe from a stranger", "is hired to restore a painting of a woman who looks exactly like them",
                 "is stranded overnight at a quiet country station", "returns home after a long war to find a different town"],
        "twist": ["Their families have been feuding for generations.", "{a} is promised to another and does not know how to say so.",
                  "A misdelivered letter makes {n} believe the worst.",
                  "A job abroad forces {n} to choose between ambition and love.",
                  "{n} hides a past that {a} would not forgive.",
                  "A sudden storm washes away the bridge and the evening plans.",
                  "A meddling aunt arranges a rival suitor.",
                  "Months pass with no word, and each assumes the other has moved on."],
        "peak": ["At the station platform, {n} runs after the departing train.", "Under a lantern-lit square, {n} finally dances with {a}.",
                 "{n} reads the last letter aloud at the wedding of someone else.", "A rainy rooftop confession sorts everything out."],
        "end": ["The two start over in a small house by the sea.", "{n} and {a} open the cafe together.",
                "They part tenderly, promising to meet in spring.", "A final kiss closes the season."],
    },
    "musical": {
        "who": ["a small-town singer", "a struggling jazz trumpeter", "a pair of vaudeville siblings", "a choir director with a failing school",
                "a sailor with a guitar", "a young dancer from the provinces", "a retired opera diva", "a travelling folk troupe"],
        "where": ["In a smoky New Orleans club", "On Broadway during the Depression", "In a village hall hosting the annual festival",
                  "At a Parisian music hall", "On a showboat drifting down the Mississippi", "In a conservatory facing closure"],
        "what": ["dreams of one big break before the theatre is demolished", "is cast by mistake in a lead role",
                 "forms a band to save the community hall", "writes a song that becomes an unlikely hit",
                 "is invited to perform for a visiting prince", "organises a talent show to pay off the landlord",
                 "loses their voice a week before the premiere", "discovers an unfinished score in the cellar"],
        "twist": ["Rehearsals collapse into rivalry between two leading ladies.", "{a}, the impresario, sells the theatre to a developer.",
                  "A scandal in the gossip pages threatens the whole production.",
                  "{n} writes a ballad about the town, and the town is not flattered.",
                  "A hurricane knocks out the power on opening night.",
                  "The orchestra goes on strike the day of the dress rehearsal.",
                  "{n} learns the dance routine one step at a time, with plenty of bruises.",
                  "A talent scout is in the audience, but nobody knows which seat."],
        "peak": ["The show goes on by candlelight, with the entire town singing along.", "{n} delivers the final song, forgetting the nerves entirely.",
                 "A show-stopping number in the rain wins over the critics.", "{n} and {a} reconcile in a duet."],
        "end": ["The theatre is saved, and the curtain falls to thunderous applause.", "{n} leaves for the big city with the whole band.",
                "The festival becomes an annual tradition.", "{n} takes a final bow under the old marquee."],
    },
    "war": {
        "who": ["a young conscript", "a field medic", "a weary sergeant", "a resistance courier", "a war correspondent",
                "a captured pilot", "a chaplain on the front line", "a codebreaker in a damp bunker"],
        "where": ["In the trenches of northern France", "In a besieged city in winter", "Behind enemy lines in occupied territory",
                  "On a destroyer in the North Atlantic", "In a prison camp in the jungle", "In a village caught between two armies"],
        "what": ["is ordered to hold a bridge nobody believes is worth holding", "carries a message that must reach headquarters by dawn",
                 "shelters a wounded enemy soldier in a cellar", "is assigned to guard a convoy of refugees",
                 "decodes a radio message that names their own unit", "survives a crash and must cross the mountains on foot",
                 "learns the offensive has been planned on faulty maps", "is left in charge when the officers are killed"],
        "twist": ["Rations run out, and the men begin to turn on each other.", "{a} questions an order that could save thousands and condemn dozens.",
                  "Shells land closer each night as the line collapses.",
                  "A frostbitten march through the forest costs {n} half the unit.",
                  "{n} receives a letter from home that changes everything.",
                  "An informer among the villagers betrays the hiding place.",
                  "The radio goes silent at the worst possible moment.",
                  "{n} has to decide whether to obey orders or to save the prisoners."],
        "peak": ["At dawn, {n} leads the last counter-attack across the open field.", "{n} destroys the bridge with the enemy halfway across.",
                 "A night crossing of the river decides the fate of the whole company.", "{n} stays behind to cover the retreat."],
        "end": ["The war ends a week later, and few of the survivors can speak of it.", "{n} returns home to a town that has forgotten.",
                "The bridge is rebuilt years later, and {n} walks across it.", "A single cross marks the place where the unit made its stand."],
    },
    "mystery": {
        "who": ["a retired inspector", "a village schoolmistress with a keen eye", "a young archivist", "a stage magician turned sleuth",
                "a gentleman detective with a limp", "a lady's maid with a notebook", "a railway clerk", "a newspaper obituary writer"],
        "where": ["In a snowbound manor house", "On a luxurious night train across the Alps", "In an English village during the annual flower show",
                  "At a lakeside hotel in the off-season", "In a private library with a locked reading room", "On a steamship crossing the Nile"],
        "what": ["is called to investigate a death that looks like an accident", "finds a clock stopped at the exact minute of a murder",
                 "is asked to find a missing will before the reading", "notices that every witness remembers the same detail wrongly",
                 "receives a coded postcard from a dead man", "is locked in with eight suspects and one corpse",
                 "discovers a second set of footprints in the snow", "is hired to find a stolen painting that was never real"],
        "twist": ["{a} has an alibi that is almost too perfect.", "A second murder proves the first suspect innocent.",
                  "Every guest hides a secret that connects them to the victim.",
                  "A torn photograph points toward a long-forgotten scandal.",
                  "The only clue is a recipe written in the margin of a ledger.",
                  "The butler insists on telling the story in the wrong order.",
                  "{n} realises the murder weapon was in plain sight all along.",
                  "A storm cuts the telegraph, leaving {n} to work alone."],
        "peak": ["In the drawing room, {n} gathers the suspects and begins to explain.", "{n} sets a trap using the missing clue as bait.",
                 "On the last night, {n} finally sees how the locked-room trick was done.", "A confession comes from the person least expected."],
        "end": ["The culprit is led away, and the guests depart in silence.", "{n} returns to the quiet village and tends the garden.",
                "The case is closed, though one question remains unanswered.", "{n} pockets the final clue and smiles."],
    },
    "fantasy": {
        "who": ["a young apprentice blacksmith", "an exiled princess", "a reluctant dragon-rider", "a lantern-bearer from the marshes",
                "a thief with a magic key", "a cursed knight", "a village herbalist", "the youngest daughter of a forgotten king"],
        "where": ["In a kingdom where the rivers sing at night", "In a forest of silver trees", "On a floating island above the clouds",
                  "In a mountain fortress guarded by ravens", "In a walled city built inside a sleeping giant", "In a desert of glass dunes"],
        "what": ["discovers a sword that whispers prophecies", "is chosen by an ancient tree to guard the last seed",
                 "must return a stolen crown to a sea witch", "finds a door in the forest that opens onto a different season",
                 "is cursed to turn to stone at sunset", "befriends a dragon hatching in a ruined tower",
                 "is told the old king's ghost can only be freed by a song", "inherits a map that rearranges itself each night"],
        "twist": ["A shape-shifting minister stirs up war between the kingdoms.", "{a}, a wandering bard, offers help for a price too high.",
                  "The magic fades a little more with each passing moon.",
                  "A frozen river must be crossed before the spring thaw.",
                  "{n} learns that the prophecy was written by the enemy.",
                  "The forest begins to close behind every step.",
                  "A council of owls demands three impossible riddles to be answered.",
                  "{a} reveals that the lost heir has been alive all along."],
        "peak": ["At the foot of the mountain, {n} faces the dragon with nothing but a song.", "{n} plants the last seed in the ashes of the burned grove.",
                 "A duel on a bridge of mist breaks the curse.", "{n} lights the great lantern and the dark recedes."],
        "end": ["The rivers sing again, and the kingdom is whole.", "{n} takes the empty throne with quiet reluctance.",
                "The dragon carries {n} home over the sunrise.", "The door in the forest closes, but the key stays in {n}'s pocket."],
    },
    "animation": {
        "who": ["a talking teapot", "a brave little tugboat", "a shy robot vacuum", "an orphaned fox kit", "a clockwork bird",
                "a family of hedgehogs", "a lonely cloud", "a paper boat with big dreams"],
        "where": ["In a toy shop that comes alive at midnight", "In an enchanted pond at the edge of the city", "In a bustling beehive town",
                  "On a mountain railway in the clouds", "In a cluttered attic full of forgotten treasures", "In a lighthouse on a tiny island"],
        "what": ["dreams of seeing the sea for the first time", "is accidentally shipped to a faraway country",
                 "sets out to find the missing winter", "hears that the toy shop will close at the end of the month",
                 "befriends a grumpy giant who never speaks", "must deliver a single letter before the first snow",
                 "wins a place in the great balloon race", "finds a map drawn in glitter"],
        "twist": ["A grumpy crow warns that the journey is too dangerous.", "{a} the squirrel is sure that {n} is the wrong sort of hero.",
                  "A thunderstorm scatters the whole group across the valley.",
                  "{n} loses the letter in a puddle and must start over.",
                  "A mischievous cat sneaks into the plan.",
                  "The bridge to the next town is only half built.",
                  "Everyone laughs at {n}'s tiny plan until it works.",
                  "A pack of toy soldiers mistakes {n} for an intruder."],
        "peak": ["With a mighty puff, {n} lifts the whole group over the waterfall.", "{n} sings a lullaby that calms the storm.",
                 "The whole town builds a bridge in one night.", "{n} makes one tiny, brave jump."],
        "end": ["Everyone gathers under the stars to celebrate.", "{n} finally sees the sea and giggles.",
                "The toy shop stays open for one more happy year.", "{n} makes a home in the lighthouse and waves at passing ships."],
    },
    "film noir": {
        "who": ["a tired private detective", "a nightclub singer with a secret", "a disgraced police lieutenant", "a corrupt city councillor's driver",
                "an insurance salesman in over his head", "a one-armed pianist", "a photographer of crime scenes", "a lawyer who knows too much"],
        "where": ["In a rain-drenched Los Angeles of the late forties", "In a neon-lit waterfront bar", "In a fog-bound harbour city",
                  "In a cheap hotel above a pawnshop", "In a city where the mayor owns the police", "In a crumbling art-deco theatre"],
        "what": ["is hired by a beautiful stranger to find her missing husband", "wakes with a gun in hand and no memory of the night before",
                 "receives a package containing half a torn photograph", "is asked to follow a woman who knows she is being followed",
                 "finds a body in the trunk of a borrowed car", "learns that the case was closed before it opened",
                 "is paid to look the other way and cannot", "meets a client who does not exist"],
        "twist": ["Every lead ends in the same dark nightclub.", "{a} lies convincingly, and {n} wants to believe her.",
                  "A corrupt captain suggests that the case should stay unsolved.",
                  "{n} is beaten in an alley and left with a warning.",
                  "The missing husband is sighted in two different cities.",
                  "A second client arrives with the same story and a different name.",
                  "A dead man's alibi falls apart in the third reel.",
                  "{n} discovers that the real target was always someone else."],
        "peak": ["On a foggy pier, {n} confronts the killer with a borrowed revolver.", "In the empty theatre, {n} lays out the truth in the dark.",
                 "A rooftop chase ends at a flashing neon sign.", "{n} hands the evidence to the one honest reporter in town."],
        "end": ["{n} walks home in the rain, a little poorer and a little wiser.", "The city goes on as it always does.",
                "{a} pays for her mistakes, and {n} pays for his.", "The last shot is an empty office and a ringing telephone."],
    },
}
# fmt: on

# Extra beats used to pad ordinary plots a little and to build the long plots that the chunker must split.
GENERIC_BEATS = [
    "Weeks pass, and the weight of the situation grows heavier for {n}.",
    "{a} offers advice that sounds kind but may hide a different purpose.",
    "In the quiet hours, {n} writes down everything that has happened, in case no one believes it later.",
    "A chance meeting with an old acquaintance forces {n} to reconsider the whole plan.",
    "Rumours spread faster than the truth, and soon even strangers know what {n} has done.",
    "Late one evening, {n} finds a small clue that had been overlooked from the beginning.",
    "The people closest to {n} begin to take sides, and some old friendships do not survive.",
    "{n} moves from one hiding place to another, always a step ahead and never certain of why.",
    "A letter arrives with news that changes what {n} thought was true about {a}.",
    "For a brief moment, it seems that everything might still be fixed with a single honest conversation.",
    "An unexpected act of kindness from a stranger reminds {n} of what is worth fighting for.",
    "Each failed attempt teaches {n} something new about the people involved.",
    "{a} tells a story from long ago that casts the whole situation in a new light.",
    "The season turns, and with it the mood of the whole community.",
    "Doubts creep in as {n} wonders whether the sacrifice will ever be understood.",
    "A sudden change in the weather forces everyone to rethink their plans.",
    "{n} is forced to make an unpopular decision and faces the consequences alone.",
    "The authorities grow suspicious and begin asking questions that {n} would rather not answer.",
    "A long night of arguments ends with a fragile truce between {n} and {a}.",
    "News of what is happening travels far beyond the town and attracts unwelcome attention.",
]

# origin -> (first names, last names, (earliest year, latest year), sampling weight)
ORIGINS: dict[str, tuple[list[str], list[str], tuple[int, int], int]] = {
    "American": (
        ["Walter", "Eleanor", "Marcus", "Dolores", "Jack", "Ruth", "Vernon", "Cora", "Dale", "Wanda"],
        [
            "Hargrove",
            "Whitaker",
            "Delacroix",
            "Pruitt",
            "Calloway",
            "Merrick",
            "Sutter",
            "Lindqvist",
            "Okafor",
            "Brennan",
        ],
        (1915, 2017),
        40,
    ),
    "British": (
        ["Edmund", "Beatrice", "Alistair", "Margery", "Neville", "Philippa", "Rupert", "Winifred", "Gethin", "Imogen"],
        [
            "Ashworth",
            "Pemberton",
            "Thackeray",
            "Fairweather",
            "Lockhart",
            "Bramwell",
            "Cholmondeley",
            "Tregenza",
            "Hollis",
            "Quayle",
        ],
        (1920, 2017),
        14,
    ),
    "Bollywood": (
        ["Arjun", "Meera", "Rohan", "Sunita", "Vikram", "Lata", "Kabir", "Anjali", "Dev", "Nandini"],
        ["Kapoor", "Mehra", "Iyer", "Chaudhary", "Bhatt", "Sengupta", "Malhotra", "Deshpande", "Rao", "Banerjee"],
        (1940, 2017),
        10,
    ),
    "Tamil": (
        ["Murugan", "Kalyani", "Senthil", "Vasanthi", "Karthik", "Lakshmi", "Pandian", "Revathi"],
        ["Rajan", "Subramaniam", "Natarajan", "Pillai", "Chandran", "Velu", "Annamalai", "Sivakumar"],
        (1950, 2017),
        5,
    ),
    "Japanese": (
        ["Haruto", "Sakura", "Daisuke", "Kimiko", "Takeshi", "Yuki", "Shinji", "Michiko"],
        ["Tanaka", "Watanabe", "Kobayashi", "Nakamura", "Ishikawa", "Fujimoto", "Hayashi", "Matsuda"],
        (1925, 2017),
        7,
    ),
    "Hong Kong": (
        ["Wai", "Mei", "Chun", "Lan", "Kwok", "Siu", "Ping", "Yee"],
        ["Chan", "Leung", "Cheung", "Lau", "Wong", "Ho", "Ng", "Yip"],
        (1960, 2017),
        4,
    ),
    "South_Korean": (
        ["Min-jun", "Seo-yeon", "Ji-ho", "Hye-jin", "Dong-hyun", "Eun-ji", "Tae-yang", "So-ra"],
        ["Kim", "Park", "Choi", "Jung", "Kang", "Yoon", "Han", "Shin"],
        (1960, 2017),
        4,
    ),
    "Australian": (
        ["Bruce", "Shelley", "Trevor", "Narelle", "Clive", "Jodie", "Gary", "Kerry"],
        ["McAllister", "Donnelly", "Kowalski", "Sinclair", "Nguyen", "Prescott", "Tully", "Fitzgerald"],
        (1970, 2017),
        4,
    ),
    "Canadian": (
        ["Gilles", "Marguerite", "Hamish", "Solange", "Duncan", "Mireille", "Lorne", "Odette"],
        ["Tremblay", "Beaulieu", "MacPherson", "Gagnon", "Sutherland", "Lavoie", "Cormier", "Whelan"],
        (1950, 2017),
        4,
    ),
    "Russian": (
        ["Dmitri", "Natalya", "Boris", "Irina", "Pavel", "Svetlana", "Anatoly", "Olga"],
        ["Volkov", "Sokolova", "Petrenko", "Morozov", "Kuznetsov", "Lebedeva", "Orlov", "Zaitsev"],
        (1925, 2017),
        4,
    ),
    "Turkish": (
        ["Emre", "Selin", "Kemal", "Aylin", "Burak", "Zeynep", "Cem", "Deniz"],
        ["Yilmaz", "Demir", "Aksoy", "Kaya", "Erdogan", "Polat", "Sahin", "Ozturk"],
        (1955, 2017),
        2,
    ),
    "Malayalam": (
        ["Gopan", "Sreedevi", "Unni", "Anitha", "Radhakrishnan", "Maya", "Vijayan", "Thankamma"],
        ["Nair", "Menon", "Varma", "Pillai", "Warrier", "Kurup", "Thampi", "Panicker"],
        (1955, 2017),
        2,
    ),
}

# Hand-written anchor films: easy targets for the demo, the UI and the evaluation set (PR-08).
ANCHORS: list[dict[str, str]] = [
    {
        "Release Year": "1947",
        "Title": "The Forgetting Hour",
        "Origin/Ethnicity": "American",
        "Director": "Marcus Hargrove",
        "Cast": "Jack Brennan, Cora Delacroix, Walter Pruitt",
        "Genre": "film noir",
        "Plot": "Detective Sam Garrity wakes up on a bench in a rain-soaked harbour district with a bullet graze on his "
        "temple and no memory of who he is or why he carries a stranger's revolver. Piecing together his own "
        "life from receipts, a hotel key and a half-burned photograph, he learns that he was hired to find a "
        "missing shipping clerk who may have seen a murder. Every witness he questions seems to recognise him, "
        "and none will say what he was like. As the trail leads to a nightclub owned by the city's most "
        "dangerous man, Garrity begins to suspect that the person he is hunting for is himself.",
    },
    {
        "Release Year": "1999",
        "Title": "Salt Lantern",
        "Origin/Ethnicity": "British",
        "Director": "Alistair Lockhart",
        "Cast": "Imogen Quayle, Edmund Ashworth, Philippa Hollis",
        "Genre": "drama",
        "Plot": "After her husband drowns in a storm, lighthouse keeper's widow Morwenna Tregarth refuses to leave the "
        "rock where she has lived for thirty years. The harbour authority plans to automate the light and "
        "send an engineer to remove her. Over a bitter winter, the young engineer and the stubborn widow "
        "begin an uneasy friendship, trading stories of the sea and of the people it has taken. When a fishing "
        "boat is lost in a blizzard, the two must keep the old lamp burning by hand through a long night.",
    },
    {
        "Release Year": "1984",
        "Title": "Orbit of the Last Orchard",
        "Origin/Ethnicity": "Russian",
        "Director": "Boris Zaitsev",
        "Cast": "Natalya Sokolova, Pavel Orlov, Irina Lebedeva",
        "Genre": "science fiction",
        "Plot": "On a failing space station the last botanist, Yelena Marchenko, tends a small orchard of apple trees "
        "that the crew regards as a harmless eccentricity. When the station's reactor starts to fail and "
        "command orders the evacuation of all biological samples, Yelena discovers that the trees have been "
        "quietly cleaning the station's air for years. She hides in the orchard and bargains with the crew: "
        "if the trees go, so does their only chance of surviving the journey home.",
    },
    {
        "Release Year": "2006",
        "Title": "The Bakery on Quill Street",
        "Origin/Ethnicity": "Canadian",
        "Director": "Solange Gagnon",
        "Cast": "Mireille Lavoie, Duncan Sutherland, Odette Whelan",
        "Genre": "comedy",
        "Plot": "When the owner of a failing Montreal bakery dies, his two feuding children must run the shop together "
        "for ninety days to inherit it. Gilles, a pompous chef, and Odette, a cynical accountant, cannot agree "
        "on a single loaf. A viral photograph of a collapsed cake, however, makes the bakery famous overnight, "
        "and customers begin queueing around the block for the worst pastries in the city. The siblings "
        "decide to lean into the disaster.",
    },
    {
        "Release Year": "1972",
        "Title": "Dust Over Cinder Creek",
        "Origin/Ethnicity": "American",
        "Director": "Dale Sutter",
        "Cast": "Vernon Calloway, Wanda Merrick, Jack Okafor",
        "Genre": "western",
        "Plot": "Sheriff Amos Reed has one week left before retirement when a stranger rides into Cinder Creek with a "
        "wounded boy and a map to a buried strongbox. The town's banker wants the map; so does a gang that has "
        "been trailing the stranger across three territories. Reed must decide whether to hand the boy over to "
        "men with legal papers or to hide him in the church bell tower and face the gang alone at sunset.",
    },
    {
        "Release Year": "2011",
        "Title": "Paper Boats in the Monsoon",
        "Origin/Ethnicity": "Bollywood",
        "Director": "Meera Sengupta",
        "Cast": "Arjun Kapoor, Anjali Mehra, Dev Iyer",
        "Genre": "romance",
        "Plot": "In Kolkata during the monsoon, a young translator named Tara receives a stack of unsigned love letters "
        "written in Bengali and accidentally delivered to her door. Determined to find the intended reader, "
        "she follows the clues through bookshops and tea stalls, and meets Rohan, a quiet printer who recognises "
        "the handwriting. As the rains intensify and the letters run out, Tara realises that the writer has "
        "been describing her all along.",
    },
]

MEMORY_LOSS_ANCHOR_TITLE = "The Forgetting Hour"
GENRE_NAMES = list(GENRES)

TITLE_ADJ = [
    "Silent",
    "Crimson",
    "Hollow",
    "Golden",
    "Last",
    "Broken",
    "Distant",
    "Burning",
    "Quiet",
    "Wandering",
    "Forgotten",
    "Iron",
    "Velvet",
    "Winter",
    "Midnight",
    "Paper",
    "Glass",
    "Hidden",
    "Restless",
    "Salt",
    "Amber",
    "Bitter",
    "Northern",
    "Copper",
]
TITLE_NOUN = [
    "River",
    "Harbour",
    "Garden",
    "Letter",
    "Season",
    "Bridge",
    "Mirror",
    "Orchard",
    "Frontier",
    "Compass",
    "Station",
    "Lantern",
    "Horizon",
    "Cathedral",
    "Promise",
    "Tide",
    "Carnival",
    "Archive",
    "Meridian",
    "Shadow",
    "Crown",
    "Engine",
    "Valley",
    "Symphony",
]
TITLE_FORMS = [
    "The {adj} {noun}",
    "{noun} of the {adj}",
    "{adj} {noun}",
    "The {noun}'s {adj} Hour",
    "A {adj} {noun}",
    "Beyond the {adj} {noun}",
]


def _name(rng: random.Random, origin: str) -> str:
    firsts, lasts, _, _ = ORIGINS[origin]
    return f"{rng.choice(firsts)} {rng.choice(lasts)}"


def _cast(rng: random.Random, origin: str) -> str:
    names: list[str] = []
    while len(names) < rng.randint(2, 4):
        candidate = _name(rng, origin)
        if candidate not in names:
            names.append(candidate)
    return ", ".join(names)


def _plot(rng: random.Random, genre: str, origin: str, *, target: str) -> str:
    """Compose a plot. ``target`` is ``short``, ``normal``, ``long``, ``edge49`` or ``edge50``."""
    g = GENRES[genre]
    n, a = _name(rng, origin), _name(rng, origin)
    while a == n:
        a = _name(rng, origin)
    who = rng.choice(g["who"])
    opening = f"{rng.choice(g['where'])}, {n}, {who}, {rng.choice(g['what'])}."
    twists = list(g["twist"])
    rng.shuffle(twists)
    if target == "short":
        parts = [opening, twists[0]]
    else:
        # "normal", "long", "edge49" and "edge50" all start from a full-length plot.
        n_twists = len(twists) if target == "long" else rng.randint(3, 5)
        parts = [opening, *twists[:n_twists]]
        beats = list(GENERIC_BEATS)
        rng.shuffle(beats)
        n_beats = 12 if target == "long" else rng.randint(1, 3)
        # Interleave generic beats after the second twist so that the climax stays at the end.
        parts = parts[:3] + beats[:n_beats] + parts[3:]
        parts += [rng.choice(g["peak"]), rng.choice(g["end"])]
    text = " ".join(parts).format(n=n, a=a)
    words = text.split()
    if target == "short":
        # Fixture rows that cleaning must drop: strictly fewer than 50 words.
        return " ".join(words[: rng.randint(14, 34)]).rstrip(",;:") + "."
    if target in EDGE_WORDS:
        # Exactly 49 words (dropped) and exactly 50 words (kept): the boundary of the cleaning rule.
        return " ".join(words[: EDGE_WORDS[target]]).rstrip(",;:.") + "."
    return text


def _title(rng: random.Random, used: set[str]) -> str:
    for _ in range(1000):
        title = rng.choice(TITLE_FORMS).format(adj=rng.choice(TITLE_ADJ), noun=rng.choice(TITLE_NOUN))
        if title not in used:
            used.add(title)
            return title
    raise RuntimeError("title space exhausted")


def _wiki(title: str) -> str:
    slug = title.replace(" ", "_")
    return f"https://en.wikipedia.org/wiki/{slug}_(synthetic_film)"


def _label_case(rng: random.Random, value: str) -> str:
    """Mimic the messy casing of the Kaggle file so that normalisation has something to do."""
    roll = rng.random()
    if roll < 0.10:
        return value.title()
    if roll < 0.12:
        return value.upper()
    return value


def build_rows() -> list[dict[str, str]]:
    rng = random.Random(SEED)
    used_titles: set[str] = {a["Title"] for a in ANCHORS}
    origins = list(ORIGINS)
    weights = [ORIGINS[o][3] for o in origins]
    rows: list[dict[str, str]] = [{**a, "Wiki Page": _wiki(a["Title"])} for a in ANCHORS]
    n_generated = N_ROWS - len(rows)
    n_fixed = N_SHORT + N_LONG + len(EDGE_WORDS)
    kinds = ["short"] * N_SHORT + ["long"] * N_LONG + list(EDGE_WORDS) + ["normal"] * (n_generated - n_fixed)
    rng.shuffle(kinds)
    for i, kind in enumerate(kinds):
        origin = rng.choices(origins, weights)[0]
        lo, hi = ORIGINS[origin][2]
        genre = GENRE_NAMES[i % len(GENRE_NAMES)] if kind != "normal" else rng.choice(GENRE_NAMES)
        title = _title(rng, used_titles)
        genre_cell = _label_case(rng, genre)
        origin_cell = origin
        if rng.random() < 0.05:
            genre_cell = "unknown"
        if rng.random() < 0.04:
            origin_cell = "Unknown"
        director = _name(rng, origin) if rng.random() > 0.05 else "Unknown"
        rows.append(
            {
                "Release Year": str(rng.randint(lo, hi)),
                "Title": title,
                "Origin/Ethnicity": origin_cell,
                "Director": director,
                "Cast": _cast(rng, origin),
                "Genre": genre_cell,
                "Wiki Page": _wiki(title),
                "Plot": _plot(rng, genre, origin, target=kind),
            }
        )
    rng.shuffle(rows)
    return rows


def write_csv(path: Path = OUTPUT) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=HEADER, lineterminator="\n")
        writer.writeheader()
        writer.writerows(build_rows())


if __name__ == "__main__":
    write_csv()
    print(f"wrote {OUTPUT}")
