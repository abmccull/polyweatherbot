const commonStyle = {
  palette: 'warm limestone, sun-bleached marble, terracotta, aged bronze, laurel green, sacred-water blue, parchment neutrals',
  artDirection: 'ancient Greek oracle sanctuary inspired by Delphi: sacred hillside terraces, solemn ritual atmosphere, premium historical strategy-game concept art',
  tone: 'reverent, mysterious, elegant, grounded, classical Hellenic, never campy or kitsch',
  materialLanguage: 'weathered marble, warm limestone, terracotta roof tile, aged bronze, carved stone, laurel leaves, incense smoke, sacred water',
  isometricCamera: 'fixed 3/4 isometric camera for a premium management sim',
  universalNegative: 'no text, no watermark, no modern objects, no logo, no sci-fi, no steampunk, no medieval castle, no Roman imperial monumentality unless explicitly requested, no Christian crosses, no bishop mitres, no Byzantine church imagery, no glossy mobile-game look, no chibi proportions, no toy-like diorama styling',
};

const terrainMaterialLanguage = 'sun-bleached limestone, weathered marble paving, dry dust, hairline cracks, restrained moss, carved terrace edges';

function joinClauses(...parts) {
  return parts.filter(Boolean).join('; ');
}

function promptBlock(...lines) {
  return lines.filter(Boolean).join('\n');
}

function terrainNegative(brief) {
  const wantsWater = /water|pool|spring|harbor|sea|river/i.test(brief);
  return joinClauses(
    commonStyle.universalNegative,
    'no gradient presentation board',
    'no disconnected floating toy blocks',
    'no candy-colored casual-mobile terrain',
    'no exaggerated bevels',
    'no props, braziers, bowls, columns, statues, roof tiles, or buildings unless explicitly requested',
    'no decorative atlas ornaments',
    wantsWater ? '' : 'no water pools, canals, or fountains'
  );
}

function portraitNegative() {
  return joinClauses(
    commonStyle.universalNegative,
    'no Christian clerical garments',
    'no papal or bishop regalia',
    'no medieval crowns unless explicitly requested',
    'no fantasy MMO styling'
  );
}

function characterNegative() {
  return joinClauses(
    commonStyle.universalNegative,
    'no fantasy MMO armor',
    'no modern fashion editorial styling',
    'no Christian or medieval ecclesiastical motifs',
    'no excessive glamour makeup'
  );
}

function buildingNegative() {
  return joinClauses(
    commonStyle.universalNegative,
    'no church nave',
    'no cathedral or chapel forms',
    'no fantasy castle towers unless explicitly requested',
    'no Roman basilica interior look',
    'no oversized imperial megastructure',
    'no sprawling civic campus',
    'no church facade language',
    'no gratuitous pools or fountains unless explicitly requested'
  );
}

function uiNegative() {
  return joinClauses(
    commonStyle.universalNegative,
    'no glossy mobile UI',
    'no neon colors',
    'no modern flat SaaS iconography'
  );
}

function asset(base) {
  return {
    status: 'planned',
    cleanupTier: 'manual_cleanup_required',
    source: 'gdd+full_release',
    ...base,
  };
}

function building(id, name, brief, options = {}) {
  return asset({
    assetId: id,
    family: 'building',
    renderProfile: options.renderProfile ?? 'building_iso_large',
    generationLane: options.generationLane ?? 'imagegen+meshy+pixel',
    wave: options.wave ?? 'precinct_expansion',
    priority: options.priority ?? 'p1',
    footprint: options.footprint ?? '2x2',
    animationProfile: options.animationProfile ?? 'ambient_fx_only',
    eraProfile: options.eraProfile ?? 'archaic,classical,hellenistic,romanized',
    variantProfile: options.variantProfile ?? 'base,damaged,upgraded',
    displayName: name,
    visualBrief: brief,
    imagePrompt: promptBlock(
      'Use case: stylized-concept',
      'Asset type: game building concept',
      `Primary request: ${name}`,
      'Scene/background: isolated single building on a plain neutral backdrop for extraction into a strategy-game asset pipeline',
      `Subject: ${brief}`,
      `Art direction: ${commonStyle.artDirection}`,
      `Historical grounding: archaic/classical Greece, Delphi-like sanctuary architecture, Hellenic sacred-site design, ${commonStyle.tone}`,
      `Style/medium: premium painted isometric game-building concept art with believable masonry and refined ornament`,
      `Composition/framing: ${commonStyle.isometricCamera}; centered single asset; full silhouette visible; compact sanctuary-scale structure; no presentation board`,
      'Lighting/mood: sun-drenched Mediterranean daylight with subtle sacred atmosphere',
      `Material language: ${commonStyle.materialLanguage}`,
      `Color palette: ${commonStyle.palette}`,
      `Constraints: ${buildingNegative()}`
    ),
    meshyPrompt: `${name}, compact ancient Greek oracle sanctuary building inspired by Delphi, ${brief}, ${commonStyle.isometricCamera}, ${commonStyle.materialLanguage}, ${commonStyle.tone}, clean silhouette, low-complexity game-ready form, isolated asset`,
    negativePrompt: buildingNegative(),
    tags: options.tags ?? 'precinct,building,oracle',
  });
}

function prop(id, name, brief, options = {}) {
  return asset({
    assetId: id,
    family: 'prop',
    renderProfile: options.renderProfile ?? 'prop_iso_small',
    generationLane: options.generationLane ?? 'meshy+pixel',
    wave: options.wave ?? 'terrain_and_props',
    priority: options.priority ?? 'p2',
    footprint: options.footprint ?? '1x1',
    animationProfile: options.animationProfile ?? 'static',
    variantProfile: options.variantProfile ?? 'base,weathered',
    displayName: name,
    visualBrief: brief,
    imagePrompt: promptBlock(
      'Use case: stylized-concept',
      'Asset type: game prop concept',
      `Primary request: ${name}`,
      'Scene/background: isolated single prop on a plain neutral backdrop for extraction',
      `Subject: ${brief}`,
      `Art direction: ${commonStyle.artDirection}`,
      `Style/medium: premium painted isometric prop concept for a historical Greek oracle sim`,
      `Composition/framing: ${commonStyle.isometricCamera}; centered with generous padding; no board layout`,
      'Lighting/mood: warm daylight with crisp readable edges and restrained sacred atmosphere',
      `Material language: ${commonStyle.materialLanguage}`,
      `Color palette: ${commonStyle.palette}`,
      `Constraints: ${buildingNegative()}`
    ),
    meshyPrompt: `${name}, sacred Greek oracle precinct prop, ${brief}, ${commonStyle.isometricCamera}, ${commonStyle.materialLanguage}, clean silhouette, isolated asset`,
    negativePrompt: buildingNegative(),
    tags: options.tags ?? 'precinct,prop,oracle',
  });
}

function terrain(id, name, brief, options = {}) {
  return asset({
    assetId: id,
    family: 'terrain',
    renderProfile: options.renderProfile ?? 'terrain_tile_set',
    generationLane: options.generationLane ?? 'imagegen+pixel',
    wave: options.wave ?? 'terrain_and_props',
    priority: options.priority ?? 'p1',
    footprint: options.footprint ?? 'tile_set',
    animationProfile: options.animationProfile ?? 'static',
    variantProfile: options.variantProfile ?? 'set',
    displayName: name,
    visualBrief: brief,
    imagePrompt: promptBlock(
      'Use case: stylized-concept',
      'Asset type: game terrain atlas reference',
      `Primary request: ${name}`,
      'Scene/background: plain neutral backdrop only; no fantasy landscape scene',
      `Subject: ${brief}`,
      `Art direction: ${commonStyle.artDirection}`,
      'Style/medium: premium painted terrain-module reference for conversion into isometric pixel-art tiles',
      'Composition/framing: clean terrain modules only; extraction-friendly spacing; no decorative props; no poster layout; no diorama scene',
      'Lighting/mood: neutral production lighting with restrained shadowing for downstream conversion',
      `Material language: ${terrainMaterialLanguage}`,
      `Color palette: ${commonStyle.palette}`,
      `Constraints: seamless extraction where applicable; ${terrainNegative(brief)}`
    ),
    meshyPrompt: '',
    negativePrompt: terrainNegative(brief),
    tags: options.tags ?? 'terrain,oracle,environment',
  });
}

function flora(id, name, brief, options = {}) {
  return asset({
    assetId: id,
    family: 'flora',
    renderProfile: options.renderProfile ?? 'prop_iso_small',
    generationLane: options.generationLane ?? 'meshy+pixel',
    wave: options.wave ?? 'terrain_and_props',
    priority: options.priority ?? 'p2',
    footprint: options.footprint ?? '1x1',
    animationProfile: options.animationProfile ?? 'ambient_sway_reference',
    variantProfile: options.variantProfile ?? 'base,seasonal',
    displayName: name,
    visualBrief: brief,
    imagePrompt: promptBlock(
      'Use case: stylized-concept',
      'Asset type: game flora concept',
      `Primary request: ${name}`,
      'Scene/background: isolated flora asset for extraction on a plain neutral backdrop',
      `Subject: ${brief}`,
      `Art direction: ${commonStyle.artDirection}`,
      'Style/medium: premium painted environment asset for a Delphi-inspired sacred precinct',
      `Composition/framing: ${commonStyle.isometricCamera}; centered single asset; no layout board`,
      'Lighting/mood: warm Greek daylight with subtle sacred atmosphere',
      `Color palette: ${commonStyle.palette}`,
      `Constraints: ${buildingNegative()}`
    ),
    meshyPrompt: `${name}, Mediterranean flora for an ancient Greek sacred precinct, ${brief}, ${commonStyle.isometricCamera}, clean silhouette`,
    negativePrompt: buildingNegative(),
    tags: options.tags ?? 'flora,oracle,environment',
  });
}

function fauna(id, name, brief, options = {}) {
  return asset({
    assetId: id,
    family: 'fauna',
    renderProfile: options.renderProfile ?? 'creature_reference',
    generationLane: options.generationLane ?? 'imagegen+meshy+pixel',
    wave: options.wave ?? 'characters_and_fauna',
    priority: options.priority ?? 'p2',
    footprint: options.footprint ?? 'creature',
    animationProfile: options.animationProfile ?? 'key_pose_reference',
    variantProfile: options.variantProfile ?? 'adult,young',
    displayName: name,
    visualBrief: brief,
    imagePrompt: promptBlock(
      'Use case: stylized-concept',
      'Asset type: game creature concept',
      `Primary request: ${name}`,
      'Scene/background: neutral backdrop for a creature turnaround or key-pose reference',
      `Subject: ${brief}`,
      `Art direction: ${commonStyle.artDirection}`,
      'Style/medium: premium creature reference sheet for a historical-mythic Greek strategy game',
      'Composition/framing: clean key-pose reference with readable silhouette and extraction-friendly spacing',
      'Lighting/mood: neutral studio lighting',
      `Color palette: ${commonStyle.palette}`,
      `Constraints: ${characterNegative()}`
    ),
    meshyPrompt: `${name}, stylized creature for an ancient Greek oracle precinct, ${brief}, clean silhouette, pose reference`,
    negativePrompt: characterNegative(),
    tags: options.tags ?? 'fauna,oracle,omen',
  });
}

function character(id, name, brief, options = {}) {
  return asset({
    assetId: id,
    family: 'character',
    renderProfile: options.renderProfile ?? 'walker_role_sheet',
    generationLane: options.generationLane ?? 'imagegen+meshy+pixel',
    wave: options.wave ?? 'characters_and_fauna',
    priority: options.priority ?? 'p1',
    footprint: options.footprint ?? 'walker',
    animationProfile: options.animationProfile ?? '8dir_walk+idle',
    variantProfile: options.variantProfile ?? 'role_sheet,portrait',
    displayName: name,
    visualBrief: brief,
    imagePrompt: promptBlock(
      'Use case: stylized-concept',
      'Asset type: game character concept',
      `Primary request: ${name}`,
      'Scene/background: plain neutral backdrop for a role sheet in a sacred Greek oracle-management sim',
      `Subject: ${brief}`,
      `Art direction: ${commonStyle.artDirection}`,
      'Historical grounding: archaic/classical Greek clothing, drapery, sandals, laurel, bronze adornment; no medieval ecclesiastical motifs',
      'Style/medium: premium character turnaround and key-pose sheet for a serious historical-fantasy management game',
      'Composition/framing: readable full-body role sheet with optional small bust inset; clean spacing; no fashion editorial layout',
      'Lighting/mood: neutral studio light with production-reference clarity',
      `Color palette: ${commonStyle.palette}`,
      `Constraints: ${characterNegative()}`
    ),
    meshyPrompt: `${name}, stylized humanoid for an ancient Greek oracle sim, ${brief}, ${commonStyle.tone}, clean silhouette, neutral pose, game-ready base mesh reference`,
    negativePrompt: characterNegative(),
    tags: options.tags ?? 'character,oracle,walker',
  });
}

function portrait(id, name, brief, options = {}) {
  return asset({
    assetId: id,
    family: 'portrait',
    renderProfile: options.renderProfile ?? 'portrait_bust',
    generationLane: options.generationLane ?? 'imagegen+pixel',
    wave: options.wave ?? 'characters_and_fauna',
    priority: options.priority ?? 'p2',
    footprint: options.footprint ?? 'portrait',
    animationProfile: options.animationProfile ?? 'expressions_3',
    variantProfile: options.variantProfile ?? 'neutral,concerned,pleased',
    displayName: name,
    visualBrief: brief,
    imagePrompt: promptBlock(
      'Use case: stylized-concept',
      'Asset type: game portrait',
      `Primary request: ${name}`,
      'Scene/background: simple neutral bust backdrop for parchment-and-bronze strategy UI integration',
      `Subject: ${brief}`,
      `Art direction: ${commonStyle.artDirection}`,
      'Historical grounding: Hellenic sacred-site society; no Christian, Byzantine, or medieval church imagery',
      'Style/medium: premium painted portrait for a historical strategy UI with restrained ornament and believable fabric/material detail',
      'Composition/framing: chest-up bust portrait with expression clarity and readable advisor silhouette',
      'Lighting/mood: warm directional light; solemn and intelligent rather than heroic-fantasy bombast',
      `Color palette: ${commonStyle.palette}`,
      `Constraints: ${portraitNegative()}`
    ),
    meshyPrompt: '',
    negativePrompt: portraitNegative(),
    tags: options.tags ?? 'portrait,ui,oracle',
  });
}

function ui(id, name, brief, options = {}) {
  return asset({
    assetId: id,
    family: 'ui',
    renderProfile: options.renderProfile ?? 'ui_pack',
    generationLane: options.generationLane ?? 'imagegen+pixel',
    wave: options.wave ?? 'world_ui',
    priority: options.priority ?? 'p2',
    footprint: options.footprint ?? 'ui_pack',
    animationProfile: options.animationProfile ?? 'static',
    variantProfile: options.variantProfile ?? 'pack',
    displayName: name,
    visualBrief: brief,
    imagePrompt: promptBlock(
      'Use case: stylized-concept',
      'Asset type: game UI icon',
      `Primary request: ${name}`,
      'Scene/background: transparent or neutral background only',
      `Subject: ${brief}`,
      `Art direction: ${commonStyle.artDirection}`,
      'Style/medium: parchment, carved-stone, bronze, and laurel strategy UI set with historical elegance',
      'Composition/framing: centered icons or emblem sheet with generous padding and extraction-friendly spacing',
      'Lighting/mood: crisp readable symbols with restrained depth',
      `Color palette: ${commonStyle.palette}`,
      `Constraints: ${uiNegative()}`
    ),
    meshyPrompt: '',
    negativePrompt: uiNegative(),
    tags: options.tags ?? 'ui,oracle,iconography',
  });
}

export const assetCatalog = [
  building('sacred_way_kit', 'Sacred Way Kit', 'processional road pieces with votive markers, braziers, low stone borders, and ceremonial paving', { wave: 'core_slice', priority: 'p0', footprint: 'tile_kit', variantProfile: 'straight,curve,t_junction,cross,festival' }),
  building('gatehouse_entrance', 'Gatehouse Entrance', 'main precinct gate with marble posts, bronze trim, and a welcoming pilgrim threshold', { wave: 'core_slice', priority: 'p1' }),
  building('castalian_spring', 'Castalian Spring', 'sacred spring basin carved into limestone with shimmering water and ritual steps', { wave: 'core_slice', priority: 'p0', footprint: '2x2' }),
  building('purification_font', 'Purification Font', 'small ritual basin and fountain where priests and pilgrims cleanse before rites', { wave: 'precinct_expansion', priority: 'p1', footprint: '1x2' }),
  building('eternal_flame_brazier', 'Eternal Flame Brazier', 'bronze ceremonial brazier on a stone plinth with sacred perpetual flame housing', { wave: 'core_slice', priority: 'p0', footprint: '1x1' }),
  building('sacrificial_altar', 'Sacrificial Altar', 'stone altar with drainage channels, offering table, and ritual implements', { wave: 'precinct_expansion', priority: 'p0', footprint: '2x2' }),
  building('inner_sanctum', 'Inner Sanctum', 'sheltered sacred chamber with omphalos plinth, vapor vents, and austere marble steps', { wave: 'core_slice', priority: 'p0', footprint: '2x2' }),
  building('omphalos_stone', 'Omphalos Stone', 'prestige monument stone wrapped in fillets and offerings, set on a sacred plinth', { wave: 'precinct_expansion', priority: 'p1', footprint: '1x1' }),
  building('temple_of_apollo', 'Temple of Apollo', 'grand marble temple with gilded pediment, laurel motifs, and sun-washed colonnade', { wave: 'precinct_expansion', priority: 'p1', footprint: '3x3' }),
  building('temple_of_athena', 'Temple of Athena', 'scholarly temple with stately columns, carved owl motifs, and scroll alcoves', { wave: 'precinct_expansion', priority: 'p2', footprint: '3x3' }),
  building('temple_of_hermes', 'Temple of Hermes', 'compact temple with traveler offerings, winged motifs, and courier niches', { wave: 'precinct_expansion', priority: 'p2', footprint: '3x3' }),
  building('tholos', 'Tholos', 'circular marble sanctuary with concentric steps, bronze roof accents, and mysterious inner glow', { wave: 'precinct_expansion', priority: 'p2', footprint: '3x3' }),
  building('treasury_vault', 'Treasury Vault', 'city-state sponsored treasury facade with heavy bronze doors and inscribed dedicatory plaques', { wave: 'precinct_expansion', priority: 'p1', footprint: '2x2' }),
  building('priest_quarters', 'Priest Quarters', 'cluster of whitewashed living quarters around a shared courtyard and shrine', { wave: 'core_slice', priority: 'p0', footprint: '2x2' }),
  building('gymnasium', 'Gymnasium', 'training court with colonnade, sand floor, exercise equipment, and shaded edge rooms', { wave: 'precinct_expansion', priority: 'p2', footprint: '3x2' }),
  building('scriptorium', 'Scriptorium', 'quiet scholarly hall with writing desks, skylight, scroll racks, and tiled floor', { wave: 'precinct_expansion', priority: 'p1', footprint: '2x2' }),
  building('animal_pen', 'Animal Pen', 'fenced enclosure with feeding troughs, shade awning, and stone gate for sacrificial stock', { wave: 'precinct_expansion', priority: 'p0', footprint: '2x2' }),
  building('granary', 'Granary', 'elevated grain store with heavy bins, weathered wood doors, and dry-stone base', { wave: 'precinct_expansion', priority: 'p1', footprint: '2x2' }),
  building('grain_field', 'Grain Field', 'terraced grain plots with low retaining walls, sheaves, irrigation channels, and harvest markers', { wave: 'precinct_expansion', priority: 'p1', footprint: '3x2', generationLane: 'blender+pixel' }),
  building('kitchen', 'Kitchen', 'working kitchen with ovens, bread tables, smoke vents, and service counters', { wave: 'precinct_expansion', priority: 'p1', footprint: '2x2' }),
  building('olive_grove', 'Olive Grove', 'sacred olive terrace with gnarled trees, collection jars, and dry-stone orchard edges', { wave: 'precinct_expansion', priority: 'p1', footprint: '3x2', generationLane: 'blender+pixel' }),
  building('olive_press', 'Olive Press', 'mechanical press house with stone press wheel, oil jars, and rustic roof', { wave: 'precinct_expansion', priority: 'p1', footprint: '2x2' }),
  building('incense_store', 'Incense Store', 'cool sealed storeroom with incense chests, hanging censers, and guarded doors', { wave: 'precinct_expansion', priority: 'p1', footprint: '2x2' }),
  building('incense_workshop', 'Incense Workshop', 'compact blending workshop with resin jars, sheltered mixing table, and perfumed braziers', { wave: 'precinct_expansion', priority: 'p1', footprint: '2x2', generationLane: 'blender+pixel' }),
  building('papyrus_reed_bed', 'Papyrus Reed Bed', 'irrigated reed basin with papyrus clusters, cut-stalk bundles, and wet stone channels', { wave: 'precinct_expansion', priority: 'p1', footprint: '3x2', generationLane: 'blender+pixel' }),
  building('storehouse', 'Storehouse', 'broad utility storehouse with amphora stacks, crate bays, and shaded loading awning', { wave: 'core_slice', priority: 'p0', footprint: '2x2' }),
  building('xenon', 'Xenon Pilgrim Inn', 'pilgrim inn with courtyard rooms, shaded arcade, and welcoming entrance banners', { wave: 'precinct_expansion', priority: 'p1', footprint: '3x2' }),
  building('agora_market', 'Agora Market', 'market plaza with striped stalls, votive vendors, and public gathering space', { wave: 'precinct_expansion', priority: 'p1', footprint: '3x2' }),
  building('theater', 'Theater', 'small hillside theater with semicircular seating and a sacred performance stage', { wave: 'precinct_expansion', priority: 'p2', footprint: '3x3' }),
  building('bath_house', 'Bath House', 'pilgrim bath complex with steam rooms, tiled basins, and bronze fixtures', { wave: 'precinct_expansion', priority: 'p2', footprint: '3x2' }),
  building('votive_offering_rack', 'Votive Offering Rack', 'dense arrangement of hanging tablets, ribbons, figurines, and donation hooks', { wave: 'precinct_expansion', priority: 'p1', footprint: '1x2' }),
  building('courier_station', 'Courier Station', 'Hermes-linked dispatch post with messenger benches, map table, and fast-horse tack', { wave: 'world_ui', priority: 'p2', footprint: '2x2' }),
  building('shadow_office', 'Shadow Office', 'discreet intelligence office tucked behind the precinct with coded shelves and hidden alcoves', { wave: 'world_ui', priority: 'p2', footprint: '2x2' }),
  building('library', 'Sacred Library', 'quiet archive library with tall shelves, ladders, scroll tubes, and bronze reading lamps', { wave: 'world_ui', priority: 'p2', footprint: '2x2' }),
  building('quarry', 'Marble Quarry', 'terraced quarry cut into the hillside with blocks, cranes, and chisel stations', { wave: 'terrain_and_props', priority: 'p2', footprint: 'landmark' }),
  building('stonemason', 'Stonemason Yard', 'open yard with marble blocks, chiseling tables, templates, and masonry sheds', { wave: 'terrain_and_props', priority: 'p2', footprint: '3x2' }),
  building('excavation_trench', 'Excavation Trench', 'ritual excavation cut exposing buried walls, ladders, and sacred debris', { wave: 'late_game', priority: 'p2', footprint: 'landmark' }),
  building('relic_vault', 'Relic Vault', 'secure underground shrine vault with relic plinths, heavy doors, and sacred lamplight', { wave: 'late_game', priority: 'p2', footprint: '2x2' }),
  building('buried_chamber', 'Buried Chamber', 'ancient hidden chamber under the precinct with collapsed masonry and mythic iconography', { wave: 'late_game', priority: 'p2', footprint: '2x2' }),

  prop('amphora_stack', 'Amphora Stack', 'clustered oil and wine amphorae tied for storage beside a wall'),
  prop('grain_sacks', 'Grain Sack Pile', 'bundled grain sacks stacked beside storage bins and scoops'),
  prop('oil_jars', 'Oil Jar Set', 'sealed ceramic jars with bronze-stoppered lids for ritual oil'),
  prop('incense_censer', 'Incense Censer', 'ornate hanging censer with filigree vents and incense ash tray'),
  prop('offering_bowl', 'Offering Bowl', 'polished bronze offering bowl with ribbons and coin offerings'),
  prop('laurel_wreath_set', 'Laurel Wreath Set', 'bundled laurel wreaths for festivals, rituals, and honored guests'),
  prop('scroll_shelves', 'Scroll Shelf Set', 'timber and bronze shelving filled with labeled papyrus rolls'),
  prop('writing_desk', 'Writing Desk', 'scholar desk with ink pots, stylus, wax tablets, and rolled scrolls'),
  prop('market_stall_set', 'Market Stall Set', 'modular market stall pieces with awnings, counters, and hanging goods', { footprint: 'kit', variantProfile: 'food,votive,luxury' }),
  prop('votive_statue_small', 'Small Votive Statue', 'dedicated marble or bronze devotional statue for pilgrims to leave behind'),
  prop('votive_statue_large', 'Large Votive Statue', 'prestige devotional statue on a low plinth with inscription plate'),
  prop('bronze_tripod', 'Bronze Tripod', 'ceremonial tripod cauldron used as prestigious sacred furnishing'),
  prop('ritual_basin', 'Ritual Basin', 'stone basin for washing hands and consecrated implements'),
  prop('stone_bench', 'Stone Bench', 'carved bench for pilgrims or waiting envoys along the processional road'),
  prop('scaffold_set', 'Scaffold Set', 'construction scaffold pieces for upgrade and repair states', { variantProfile: 'light,heavy' }),
  prop('repair_tools', 'Repair Tool Set', 'custodian tool bundle with mallet, rope, bucket, and chisels'),
  prop('treasury_chest', 'Treasury Chest', 'heavy coffer with bronze banding and secure ceremonial lock'),
  prop('festival_banner_set', 'Festival Banner Set', 'fabric banners and streamers for Panhellenic festival decoration', { variantProfile: 'apollo,athena,hermes,neutral' }),
  prop('torch_post', 'Torch Post', 'tall torch post with bronze cup and carved stone base'),
  prop('marble_column_fragment', 'Marble Column Fragment', 'broken column drums and carved capitals for ruins or excavation dressing'),
  prop('sacred_boundary_marker', 'Sacred Boundary Marker', 'small carved marker stone denoting holy ground'),
  prop('sacrifice_table', 'Sacrifice Table', 'sturdy ritual table with channels, knives, bowls, and linen cloth'),
  prop('caravan_cart', 'Caravan Cart', 'two-wheeled goods cart with sacks, amphorae, and travel gear'),
  prop('archive_scroll_crate', 'Archive Scroll Crate', 'crate packed with sealed papyrus rolls and wax labels'),
  prop('debate_podium', 'Debate Podium', 'portable speaker platform for philosophers and public disputation'),

  terrain('limestone_terrace_tiles', 'Limestone Terrace Tiles', 'dry sun-bleached limestone terrace ground modules only, with subtle wear and no props or water', { wave: 'core_slice', priority: 'p0' }),
  terrain('dry_earth_tiles', 'Dry Earth Tiles', 'dusty sun-baked earth tiles for open hillside spaces', { wave: 'core_slice', priority: 'p0' }),
  terrain('sacred_paving_tiles', 'Sacred Paving Tiles', 'finer ceremonial paving tiles with inset geometric patterns', { wave: 'core_slice', priority: 'p0' }),
  terrain('marble_platform_tiles', 'Marble Platform Tiles', 'prestige marble platform tile set for temples and sanctum terraces', { wave: 'precinct_expansion', priority: 'p1' }),
  terrain('cliff_edge_set', 'Cliff Edge Set', 'dry stepped limestone cliff-edge modules and corners only for the rising hillside precinct, with no props or decorative structures', { wave: 'core_slice', priority: 'p0', variantProfile: 'north,south,inner_corner,outer_corner' }),
  terrain('stone_stair_set', 'Stone Stair Set', 'plain limestone stair modules only for climbing sacred terraces, with no props or altars', { wave: 'core_slice', priority: 'p0' }),
  terrain('retaining_wall_set', 'Retaining Wall Set', 'dry-stone and marble retaining wall pieces for terracing', { wave: 'core_slice', priority: 'p1' }),
  terrain('spring_pool_tiles', 'Spring Pool Tiles', 'water-edge tiles and stone-lipped sacred pool pieces', { wave: 'core_slice', priority: 'p1' }),
  terrain('excavation_rubble_tiles', 'Excavation Rubble Tiles', 'broken masonry, dirt, and shard-strewn tiles for digs and ruins', { wave: 'late_game', priority: 'p2' }),
  terrain('quarry_cut_tiles', 'Quarry Cut Tiles', 'rough-hewn marble cut tiles and quarry debris ground set', { wave: 'terrain_and_props', priority: 'p2' }),
  terrain('sacred_shadow_decals', 'Sacred Shadow Decals', 'soft ground shadows and wear decals for processions, columns, and clutter', { wave: 'terrain_and_props', priority: 'p2', generationLane: 'pixel', renderProfile: 'decal_pack' }),

  flora('olive_tree_mature', 'Mature Olive Tree', 'gnarled mature olive tree with silvery leaves and twisted trunk', { wave: 'core_slice', priority: 'p1', footprint: '2x2' }),
  flora('olive_tree_young', 'Young Olive Tree', 'slimmer cultivated olive tree for groves and precinct edges', { wave: 'terrain_and_props', priority: 'p2' }),
  flora('cypress_tree', 'Cypress Tree', 'tall dark cypress tree used as a sacred vertical accent', { wave: 'terrain_and_props', priority: 'p1', footprint: '1x2' }),
  flora('laurel_shrub', 'Laurel Shrub', 'dense laurel shrub for ritual groves and sanctified corners', { wave: 'terrain_and_props', priority: 'p2' }),
  flora('dry_grass_cluster', 'Dry Grass Cluster', 'tufts of hardy golden grass for rocky edges and sparse terraces', { wave: 'terrain_and_props', priority: 'p2', generationLane: 'imagegen+pixel' }),
  flora('rocky_outcrop', 'Rocky Outcrop', 'sun-baked limestone rocks with crisp silhouette for map edges', { wave: 'terrain_and_props', priority: 'p1', footprint: '2x1' }),

  fauna('goat', 'Goat', 'nimble sacred-site goat with short horns and dusty white-brown coat', { wave: 'precinct_expansion', priority: 'p1' }),
  fauna('sheep', 'Sheep', 'woolly sacrificial sheep suited for pens and ritual procession', { wave: 'precinct_expansion', priority: 'p1' }),
  fauna('ox', 'Ox', 'broad slow-moving ox used for prestige sacrifices and heavy hauling', { wave: 'precinct_expansion', priority: 'p2' }),
  fauna('white_dove', 'White Dove', 'small ceremonial dove used in sacred iconography and ambient life', { wave: 'characters_and_fauna', priority: 'p2', generationLane: 'imagegen+pixel' }),
  fauna('raven', 'Raven', 'dark omen bird with glossy feathers and sharp profile', { wave: 'characters_and_fauna', priority: 'p2', generationLane: 'imagegen+pixel' }),
  fauna('eagle', 'Eagle', 'regal eagle used for high-prestige omens and consultation illustration', { wave: 'characters_and_fauna', priority: 'p2', generationLane: 'imagegen+pixel' }),
  fauna('serpent', 'Serpent', 'coiling serpent motif creature for dreams, rituals, and omen scenes', { wave: 'characters_and_fauna', priority: 'p2', generationLane: 'imagegen+pixel' }),
  fauna('owl', 'Owl', 'watchful owl tied to scholarly and night omen imagery', { wave: 'characters_and_fauna', priority: 'p2', generationLane: 'imagegen+pixel' }),

  character('pythia_main', 'Pythia — Ceremonial Base', 'Delphic chief prophetess in layered Hellenic ritual robes, laurel crown, sacred jewelry, and solemn ceremonial bearing', { wave: 'core_slice', priority: 'p0', renderProfile: 'pythia_sheet', animationProfile: 'ritual_pose+idle', variantProfile: 'calm,trance,spent' }),
  character('pythia_elder', 'Pythia — Elder Variant', 'aged legendary prophetess with regal bearing, fragile body, and immense authority', { wave: 'late_game', priority: 'p2', renderProfile: 'pythia_sheet', animationProfile: 'ritual_pose+idle', variantProfile: 'calm,trance,ailing' }),
  character('priest_attendant', 'Priest Attendant', 'white-robed temple attendant carrying ritual cloths and water vessels', { wave: 'core_slice', priority: 'p0' }),
  character('augur', 'Augur', 'bird-reading priest with staff, satchel, and watchful upward posture', { wave: 'core_slice', priority: 'p1' }),
  character('spring_warden', 'Spring Warden', 'water-keeper priest with simple robes, basin tools, and measured calm demeanor', { wave: 'core_slice', priority: 'p0' }),
  character('flame_keeper', 'Flame Keeper', 'guardian of the sacred flame carrying oil and ember tools', { wave: 'core_slice', priority: 'p0' }),
  character('sacrificial_priest', 'Sacrificial Priest', 'blood-rite officiant in layered apron robes with ritual knife and bowl', { wave: 'precinct_expansion', priority: 'p1' }),
  character('dream_priest', 'Dream Priest', 'mystic priest with sleep chamber accessories and incense bundles', { wave: 'characters_and_fauna', priority: 'p2' }),
  character('astronomer', 'Astronomer', 'sky-watching scholar in dark indigo robes with charts and bronze sighting tools', { wave: 'characters_and_fauna', priority: 'p2' }),
  character('scholar', 'Scholar', 'literate scholar in Athena-aligned robes carrying papyrus and wax tablets', { wave: 'precinct_expansion', priority: 'p1' }),
  character('diplomat_priest', 'Diplomat Priest', 'smooth ceremonial official balancing piety with worldly grace', { wave: 'world_ui', priority: 'p2' }),
  character('spy_handler', 'Shadow Handler', 'quiet intelligence operative in discreet dark robes with hidden documents', { wave: 'world_ui', priority: 'p2' }),
  character('custodian', 'Custodian', 'maintenance worker with rope, mallet, and patched practical tunic', { wave: 'core_slice', priority: 'p1' }),
  character('carrier', 'Resource Carrier', 'precinct hauler carrying baskets, amphorae, and practical travel wear', { wave: 'core_slice', priority: 'p1' }),
  character('builder_worker', 'Builder Worker', 'stonemason laborer with chisels, dusted tunic, and strong silhouette', { wave: 'terrain_and_props', priority: 'p2' }),
  character('ordinary_pilgrim', 'Ordinary Pilgrim', 'humble pilgrim in travel-worn garments bearing small offerings', { wave: 'core_slice', priority: 'p0' }),
  character('wealthy_pilgrim', 'Wealthy Pilgrim', 'better-dressed pilgrim with jewelry, fine fabric, and premium offerings', { wave: 'precinct_expansion', priority: 'p1' }),
  character('sick_pilgrim', 'Sick Pilgrim', 'ailing pilgrim moving carefully with wrappings and anxious posture', { wave: 'characters_and_fauna', priority: 'p2' }),
  character('festival_pilgrim', 'Festival Pilgrim', 'cheerful pilgrim in brighter cloth with ribbons and celebratory offerings', { wave: 'precinct_expansion', priority: 'p2' }),
  character('envoy_military', 'Military Envoy', 'formal envoy with cuirass elements, cloak, and retinue authority', { wave: 'core_slice', priority: 'p1' }),
  character('envoy_mercantile', 'Mercantile Envoy', 'well-appointed commercial envoy with scroll case and rich travel wear', { wave: 'world_ui', priority: 'p2' }),
  character('envoy_spiritual', 'Spiritual Envoy', 'pious embassy representative with sacred gifts and measured reverence', { wave: 'world_ui', priority: 'p2' }),
  character('envoy_imperial', 'Imperial Envoy', 'high-status late-game envoy with formal regalia and intimidating retinue presence', { wave: 'late_game', priority: 'p2' }),
  character('rationalist_philosopher', 'Rationalist Philosopher', 'sharp-featured public thinker in austere robes arguing against supernatural claims', { wave: 'world_ui', priority: 'p2' }),
  character('moralist_philosopher', 'Moralist Philosopher', 'earnest critic condemning corruption and political compromise', { wave: 'world_ui', priority: 'p2' }),
  character('rival_theologian', 'Rival Theologian', 'competing sacred authority in ornate but antagonistic priestly dress', { wave: 'world_ui', priority: 'p2' }),
  character('political_agitator', 'Political Agitator', 'fiery public speaker with faction colors and manipulative charisma', { wave: 'world_ui', priority: 'p2' }),
  character('corrupt_priest', 'Corrupt Priest', 'insider priest whose public piety masks subtle greed and compromise', { wave: 'world_ui', priority: 'p2' }),
  character('rival_oracle_master', 'Rival Oracle Master', 'charismatic leader of a competing oracle with dangerous prestige', { wave: 'late_game', priority: 'p2' }),
  character('saboteur', 'Saboteur Agent', 'covert operative in travel cloak carrying hidden tools and false papers', { wave: 'late_game', priority: 'p2' }),

  portrait('hierophant_portrait', 'Hierophant Portrait', 'grave Hellenic hierophant with laurel fillet, sacred staff, ritual authority, and no Christian regalia', { wave: 'core_slice', priority: 'p1' }),
  portrait('treasurer_portrait', 'Treasurer Portrait', 'anxious brilliant Greek treasurer with scrolls, tablets, abacus-like counting tools, and restrained wealth', { wave: 'core_slice', priority: 'p1' }),
  portrait('builder_portrait', 'Master Builder Portrait', 'gruff veteran Greek master builder with stone dust, measured confidence, and practical craft attire', { wave: 'core_slice', priority: 'p1' }),
  portrait('diplomat_portrait', 'Diplomat Portrait', 'smooth worldly Greek diplomat in refined but restrained classical dress with intelligent watchfulness', { wave: 'core_slice', priority: 'p1' }),
  portrait('shadow_portrait', 'Shadow Portrait', 'quiet intelligence chief in discreet Hellenic attire emerging from soft shadow with sharp observant eyes', { wave: 'core_slice', priority: 'p1' }),
  portrait('croesus_portrait', 'Croesus Portrait', 'wealthy king portrait with opulent Lydian richness and calculating expression', { wave: 'late_game', priority: 'p2' }),
  portrait('alexander_portrait', 'Alexander Portrait', 'young conqueror portrait with ambition, charisma, and imperial tension', { wave: 'late_game', priority: 'p2' }),
  portrait('roman_envoy_portrait', 'Roman Envoy Portrait', 'stern late-era Roman diplomat portrait with pragmatic authority', { wave: 'late_game', priority: 'p2' }),

  ui('faction_banner_pack', 'Faction Banner Pack', 'banner set for city-states and patrons with faction motifs and heraldic sacred styling', { wave: 'world_ui', priority: 'p1', variantProfile: '12 faction variants' }),
  ui('city_state_marker_pack', 'City-State Marker Pack', 'map markers for city-states, capitals, minor allies, and patron sites', { wave: 'world_ui', priority: 'p1', variantProfile: 'city,capital,ally,hostile' }),
  ui('trade_route_marker_pack', 'Trade Route Marker Pack', 'map overlays for trade caravans, sea trade, blockades, and embargoes', { wave: 'world_ui', priority: 'p2', variantProfile: 'land,sea,embargo,bonus' }),
  ui('war_front_marker_pack', 'War Front Marker Pack', 'military conflict overlays for raids, sieges, pitched battles, and musters', { wave: 'world_ui', priority: 'p2', variantProfile: 'raid,siege,battle,muster' }),
  ui('philosopher_threat_pack', 'Philosopher Threat Pack', 'icons and callouts for philosopher threat types and public debates', { wave: 'world_ui', priority: 'p2', variantProfile: 'rationalist,moralist,theologian,agitator' }),
  ui('rival_oracle_marker_pack', 'Rival Oracle Marker Pack', 'world and precinct markers for rival sites, rumors, and sabotage alerts', { wave: 'world_ui', priority: 'p2', variantProfile: 'site,rumor,sabotage,duel' }),
  ui('treasury_dedication_pack', 'Treasury Dedication Pack', 'icons and seals for city-state treasury dedications and patron gifts', { wave: 'world_ui', priority: 'p2', variantProfile: 'gift,treasury,monument,seal' }),
  ui('age_emblem_pack', 'Age Emblem Pack', 'decorative emblems for Archaic, Classical, Hellenic, Hellenistic, and Roman Shadow eras', { wave: 'late_game', priority: 'p2', variantProfile: '5 age variants' }),
  ui('origin_emblem_pack', 'Origin Emblem Pack', 'origin emblems for Ancient Spring, Upstart Shrine, Cursed Oracle, War Oracle, God\'s Favourite, Merchant Oracle, and Exile\'s Oracle', { wave: 'late_game', priority: 'p2', variantProfile: '7 origin variants' }),
  ui('omen_icon_pack', 'Omen Icon Pack', 'omen icon family covering ornithomancy, pyromancy, hydromancy, extispicy, oneiromancy, astromancy, and chresmology', { wave: 'core_slice', priority: 'p1', variantProfile: '7 omen variants' }),
  ui('consultation_ornament_pack', 'Consultation Ornament Pack', 'laurel wreaths, incense borders, carved stone headers, and sacred seals for consultation UI', { wave: 'core_slice', priority: 'p1', variantProfile: 'frame,seal,divider,header' }),
];
