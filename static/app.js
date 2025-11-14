const state = {
    featureOrder: [],
    models: [],
    classLabels: [],
    classInfo: [],
    dataset: null,
    targetColumn: null,
};

const selectors = {
    individualForm: () => document.getElementById('individual-form'),
    individualModel: () => document.getElementById('individual-model'),
    featureInputs: () => document.getElementById('feature-inputs'),
    individualStatus: () => document.getElementById('individual-status'),
    individualResult: () => document.getElementById('individual-result'),
    batchForm: () => document.getElementById('batch-form'),
    batchModel: () => document.getElementById('batch-model'),
    batchStatus: () => document.getElementById('batch-status'),
    batchResult: () => document.getElementById('batch-result'),
    batchMetrics: () => document.getElementById('batch-metrics'),
    confusionImage: () => document.getElementById('confusion-image'),
    confusionTable: () => document.getElementById('confusion-table'),
    batchPreview: () => document.getElementById('batch-preview'),
    referenceContent: () => document.getElementById('reference-content'),
    heroClassCount: () => document.getElementById('hero-class-count'),
    heroFeatureCount: () => document.getElementById('hero-feature-count'),
    tabButtons: () => document.querySelectorAll('[data-tab-target]'),
    tabViews: () => document.querySelectorAll('[data-tab-content]'),
};

const featureTranslations = {
    male: 'Masculino',
    female: 'Femenino',
    age: 'Edad',
    urban_origin: 'Procedencia urbana',
    rural_origin: 'Procedencia rural',
    homemaker: 'Ama de casa',
    student: 'Estudiante',
    professional: 'Profesional',
    merchant: 'Comerciante',
    agriculture_livestock: 'Agropecuario',
    various_jobs: 'Diversos trabajos',
    unemployed: 'Desempleado',
    hospitalization_days: 'Dias de hospitalizacion',
    body_temperature: 'Temperatura corporal',
    fever: 'Fiebre',
    headache: 'Dolor de cabeza',
    dizziness: 'Mareo',
    loss_of_appetite: 'Perdida de apetito',
    weakness: 'Debilidad',
    myalgias: 'Mialgias',
    arthralgias: 'Artralgias',
    eye_pain: 'Dolor ocular',
    hemorrhages: 'Hemorragias',
    vomiting: 'Vomito',
    abdominal_pain: 'Dolor abdominal',
    chills: 'Escalofrios',
    hemoptysis: 'Hemoptisis',
    edema: 'Edema',
    jaundice: 'Ictericia',
    bruises: 'Moretones',
    petechiae: 'Petequias',
    rash: 'Erupcion',
    diarrhea: 'Diarrea',
    respiratory_difficulty: 'Dificultad respiratoria',
    itching: 'Comezon',
    hematocrit: 'Hematocrito',
    hemoglobin: 'Hemoglobina',
    red_blood_cells: 'Globulos rojos',
    white_blood_cells: 'Globulos blancos',
    neutrophils: 'Neutrofilos',
    eosinophils: 'Eosinofilos',
    basophils: 'Basofilos',
    monocytes: 'Monocitos',
    lymphocytes: 'Linfocitos',
    platelets: 'Plaquetas',
    'AST (SGOT)': 'AST (SGOT)',
    'ALT (SGPT)': 'ALT (SGPT)',
    'ALP (alkaline_phosphatase)': 'ALP (fosfatasa alcalina)',
    total_bilirubin: 'Bilirrubina total',
    direct_bilirubin: 'Bilirrubina directa',
    indirect_bilirubin: 'Bilirrubina indirecta',
    total_proteins: 'Proteinas totales',
    albumin: 'Albumina',
    creatinine: 'Creatinina',
    urea: 'Urea',
};

const featureCategoryDescriptions = {
    'Contexto del paciente': 'Datos base para ubicar el caso clinico.',
    'Signos vitales': 'Mediciones iniciales registradas durante la atencion.',
    'Hallazgos clinicos': 'Manifestaciones observadas durante la exploracion.',
    'Biometria hematica': 'Resultados del hemograma y formula leucocitaria.',
    'Perfil hepatico': 'Marcadores hepaticos y bilirrubinas asociadas.',
    'Funcion renal y metabolica': 'Parametros metabolicos relacionados con rinon y proteinas.',
    'Parametros de laboratorio': 'Valores cuantitativos registrados para el analisis.',
};

const featureDetails = {
    age: {
        category: 'Contexto del paciente',
        helper: 'Anos cumplidos del paciente.',
        placeholder: 'Ej: 34',
        min: 0,
        max: 110,
    },
    hospitalization_days: {
        category: 'Contexto del paciente',
        helper: 'Dias transcurridos desde el ingreso o consulta.',
        placeholder: 'Ej: 3',
        min: 0,
    },
    body_temperature: {
        category: 'Signos vitales',
        helper: 'Temperatura corporal en grados Celsius.',
        placeholder: 'Ej: 37.2',
        min: 30,
        max: 45,
        step: 0.1,
    },
    petechiae: {
        category: 'Hallazgos clinicos',
        helper: 'Conteo observado de petequias (usa 0 si no aplica).',
        placeholder: 'Ej: 0',
        min: 0,
    },
    hematocrit: {
        category: 'Biometria hematica',
        helper: 'Porcentaje de hematocrito.',
        placeholder: 'Ej: 42.5',
        min: 0,
    },
    hemoglobin: {
        category: 'Biometria hematica',
        helper: 'Concentracion de hemoglobina en g/dL.',
        placeholder: 'Ej: 13.8',
        min: 0,
    },
    red_blood_cells: {
        category: 'Biometria hematica',
        helper: 'Conteo de globulos rojos (millones/uL).',
        placeholder: 'Ej: 4.6',
        min: 0,
    },
    white_blood_cells: {
        category: 'Biometria hematica',
        helper: 'Conteo de globulos blancos (x10^3/uL).',
        placeholder: 'Ej: 6.5',
        min: 0,
    },
    neutrophils: {
        category: 'Biometria hematica',
        helper: 'Porcentaje de neutrofilos.',
        placeholder: 'Ej: 55',
        min: 0,
    },
    eosinophils: {
        category: 'Biometria hematica',
        helper: 'Porcentaje de eosinofilos.',
        placeholder: 'Ej: 2',
        min: 0,
    },
    basophils: {
        category: 'Biometria hematica',
        helper: 'Porcentaje de basofilos.',
        placeholder: 'Ej: 1',
        min: 0,
    },
    monocytes: {
        category: 'Biometria hematica',
        helper: 'Porcentaje de monocitos.',
        placeholder: 'Ej: 6',
        min: 0,
    },
    lymphocytes: {
        category: 'Biometria hematica',
        helper: 'Porcentaje de linfocitos.',
        placeholder: 'Ej: 30',
        min: 0,
    },
    platelets: {
        category: 'Biometria hematica',
        helper: 'Conteo de plaquetas (x10^3/uL).',
        placeholder: 'Ej: 220',
        min: 0,
    },
    'AST (SGOT)': {
        category: 'Perfil hepatico',
        helper: 'Actividad enzimatica AST en U/L.',
        placeholder: 'Ej: 28',
        min: 0,
    },
    'ALT (SGPT)': {
        category: 'Perfil hepatico',
        helper: 'Actividad enzimatica ALT en U/L.',
        placeholder: 'Ej: 32',
        min: 0,
    },
    'ALP (alkaline_phosphatase)': {
        category: 'Perfil hepatico',
        helper: 'Nivel de fosfatasa alcalina en U/L.',
        placeholder: 'Ej: 110',
        min: 0,
    },
    total_bilirubin: {
        category: 'Perfil hepatico',
        helper: 'Bilirrubina total en mg/dL.',
        placeholder: 'Ej: 0.9',
        min: 0,
    },
    direct_bilirubin: {
        category: 'Perfil hepatico',
        helper: 'Fraccion directa en mg/dL.',
        placeholder: 'Ej: 0.2',
        min: 0,
    },
    indirect_bilirubin: {
        category: 'Perfil hepatico',
        helper: 'Fraccion indirecta en mg/dL.',
        placeholder: 'Ej: 0.7',
        min: 0,
    },
    total_proteins: {
        category: 'Funcion renal y metabolica',
        helper: 'Proteinas totales en g/dL.',
        placeholder: 'Ej: 7.1',
        min: 0,
    },
    albumin: {
        category: 'Funcion renal y metabolica',
        helper: 'Concentracion de albumina en g/dL.',
        placeholder: 'Ej: 4.1',
        min: 0,
    },
    creatinine: {
        category: 'Funcion renal y metabolica',
        helper: 'Creatinina serica en mg/dL.',
        placeholder: 'Ej: 1.0',
        min: 0,
    },
    urea: {
        category: 'Funcion renal y metabolica',
        helper: 'Urea serica en mg/dL.',
        placeholder: 'Ej: 28',
        min: 0,
    },
};

function createOneHotValues(selectedKey, featureKeys) {
    const values = {};
    featureKeys.forEach((key) => {
        values[key] = key === selectedKey ? 1 : 0;
    });
    return values;
}

const OCCUPATION_FEATURES = [
    'homemaker',
    'student',
    'professional',
    'merchant',
    'agriculture_livestock',
    'unemployed',
    'various_jobs',
];

const groupedFeatureConfigs = [
    {
        id: 'gender',
        label: 'Genero',
        selectId: 'gender-select',
        options: [
            { value: 'male', label: 'Masculino', featureValues: { male: 1, female: 0 } },
            { value: 'female', label: 'Femenino', featureValues: { male: 0, female: 1 } },
        ],
    },
    {
        id: 'origin',
        label: 'Procedencia',
        selectId: 'origin-select',
        options: [
            {
                value: 'rural_origin',
                label: 'Procedencia rural',
                featureValues: { rural_origin: 1, urban_origin: 0 },
            },
            {
                value: 'urban_origin',
                label: 'Procedencia urbana',
                featureValues: { rural_origin: 0, urban_origin: 1 },
            },
        ],
    },
    {
        id: 'occupation',
        label: 'Ocupacion',
        selectId: 'occupation-select',
        options: [
            {
                value: 'homemaker',
                label: 'Ama de casa',
                featureValues: createOneHotValues('homemaker', OCCUPATION_FEATURES),
            },
            {
                value: 'student',
                label: 'Estudiante',
                featureValues: createOneHotValues('student', OCCUPATION_FEATURES),
            },
            {
                value: 'professional',
                label: 'Profesional',
                featureValues: createOneHotValues('professional', OCCUPATION_FEATURES),
            },
            {
                value: 'merchant',
                label: 'Comerciante',
                featureValues: createOneHotValues('merchant', OCCUPATION_FEATURES),
            },
            {
                value: 'agriculture_livestock',
                label: 'Agropecuario',
                featureValues: createOneHotValues('agriculture_livestock', OCCUPATION_FEATURES),
            },
            {
                value: 'unemployed',
                label: 'Desempleado',
                featureValues: createOneHotValues('unemployed', OCCUPATION_FEATURES),
            },
        ],
    },
];

groupedFeatureConfigs.forEach((config) => {
    const keys = new Set();
    config.options.forEach((option) => {
        Object.keys(option.featureValues).forEach((key) => keys.add(key));
    });
    config.featureKeys = Array.from(keys);
});

const groupedFeatureKeySet = new Set(
    groupedFeatureConfigs.flatMap((config) => config.featureKeys)
);

const SYMPTOM_CONFIGS = [
    { key: 'fever', label: 'Fiebre' },
    { key: 'headache', label: 'Dolor de cabeza' },
    { key: 'dizziness', label: 'Mareo' },
    { key: 'loss_of_appetite', label: 'Perdida de apetito' },
    { key: 'weakness', label: 'Debilidad' },
    { key: 'eye_pain', label: 'Dolor ocular' },
    { key: 'hemorrhages', label: 'Hemorragias' },
    { key: 'arthralgias', label: 'Artralgias' },
    { key: 'vomiting', label: 'Vomito' },
    { key: 'myalgias', label: 'Mialgias' },
    { key: 'abdominal_pain', label: 'Dolor abdominal' },
    { key: 'chills', label: 'Escalofrios' },
    { key: 'hemoptysis', label: 'Hemoptisis' },
    { key: 'edema', label: 'Edema' },
    { key: 'jaundice', label: 'Ictericia' },
    { key: 'bruises', label: 'Moretones' },
    { key: 'rash', label: 'Erupcion' },
    { key: 'diarrhea', label: 'Diarrea' },
    { key: 'respiratory_difficulty', label: 'Dificultad respiratoria' },
    { key: 'itching', label: 'Comezon' },
];

const SYMPTOM_FEATURE_SET = new Set(SYMPTOM_CONFIGS.map((item) => item.key));

const SKIPPED_INPUT_FEATURES = new Set([
    ...groupedFeatureKeySet,
    ...SYMPTOM_FEATURE_SET,
]);

function getFeatureLabel(feature) {
    if (featureTranslations[feature]) {
        return featureTranslations[feature];
    }
    return feature
        .split('_')
        .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

function setStatus(element, message, type = '') {
    const el = typeof element === 'function' ? element() : element;
    if (!el) return;
    el.textContent = message;
    el.classList.remove('error', 'success');
    if (type) {
        el.classList.add(type);
    }
}

function toggleHidden(element, hidden) {
    const el = typeof element === 'function' ? element() : element;
    if (!el) return;
    if (hidden) {
        el.classList.add('hidden');
    } else {
        el.classList.remove('hidden');
    }
}

function activateTab(tabId) {
    const buttons = Array.from(selectors.tabButtons?.() ?? []);
    const views = Array.from(selectors.tabViews?.() ?? []);
    if (buttons.length === 0 || views.length === 0) {
        return null;
    }

    let targetId = tabId;
    const hasMatch = buttons.some((button) => button.dataset.tabTarget === tabId);
    if (!hasMatch) {
        targetId = buttons[0]?.dataset.tabTarget;
    }

    buttons.forEach((button) => {
        const isActive = button.dataset.tabTarget === targetId;
        button.classList.toggle('active', isActive);
        button.setAttribute('aria-selected', isActive ? 'true' : 'false');
        button.setAttribute('tabindex', isActive ? '0' : '-1');
    });

    views.forEach((view) => {
        const isActive = view.dataset.tabContent === targetId;
        view.classList.toggle('active', isActive);
        if (isActive) {
            view.removeAttribute('hidden');
        } else {
            view.setAttribute('hidden', 'true');
        }
    });

    return targetId;
}

function tabHashToId(hash) {
    switch (hash) {
        case '#single-lab':
            return 'individual';
        case '#batch-lab':
            return 'batch';
        default:
            return null;
    }
}

function syncTabWithHash() {
    const hash = window.location.hash;
    const mapped = tabHashToId(hash);
    const activeId = activateTab(mapped ?? 'batch');
    if (hash && mapped) {
        const section = document.querySelector(hash);
        if (section) {
            section.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }
    return activeId;
}

function initTabs() {
    const buttons = Array.from(selectors.tabButtons?.() ?? []);
    if (buttons.length === 0) return;

    buttons.forEach((button) => {
        button.addEventListener('click', (event) => {
            const targetId = event.currentTarget.dataset.tabTarget;
            const activeId = activateTab(targetId ?? 'batch');
            const sectionId = activeId === 'individual' ? 'single-lab' : 'batch-lab';
            const section = document.getElementById(sectionId);
            if (section) {
                section.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
            if (sectionId) {
                if (typeof history.replaceState === 'function') {
                    history.replaceState(null, '', `#${sectionId}`);
                } else {
                    window.location.hash = sectionId;
                }
            }
        });
    });

    window.addEventListener('hashchange', syncTabWithHash);
    syncTabWithHash();
}

async function loadStatus() {
    try {
        const response = await fetch('/api/status');
        if (!response.ok) {
            throw new Error('No se pudo obtener la configuracion inicial.');
        }
        const payload = await response.json();
        state.featureOrder = payload.feature_order || [];
        state.models = payload.models || [];
        state.classInfo = Array.isArray(payload.class_labels) ? payload.class_labels : [];
        state.classLabels = state.classInfo.map((item) => String(item?.name ?? item?.id ?? item));
        state.dataset = payload.dataset || null;
        state.targetColumn = payload.dataset?.target_column || null;

        populateModelOptions(selectors.individualModel());
        populateModelOptions(selectors.batchModel());
        renderFeatureInputs();
        renderReferenceMetrics();
        updateHeroStats();
        setStatus(selectors.individualStatus, 'Terminal lista para predecir.', 'success');
        setStatus(selectors.batchStatus, 'Laboratorio listo para procesar.', 'success');
    } catch (error) {
        console.error(error);
        setStatus(selectors.individualStatus, 'Fallo al sincronizar la configuracion.', 'error');
        setStatus(selectors.batchStatus, 'Fallo al sincronizar la configuracion.', 'error');
    }
}

function populateModelOptions(selectElement) {
    if (!selectElement) return;
    selectElement.innerHTML = '';
    state.models.forEach((model) => {
        const option = document.createElement('option');
        option.value = model.id;
        option.textContent = model.label;
        selectElement.appendChild(option);
    });
}

function renderFeatureInputs() {
    const container = selectors.featureInputs();
    if (!container) return;
    container.innerHTML = '';
    renderGroupedControls(container);
    renderSymptomControls(container);
    const defaultCategory = 'Parametros de laboratorio';
    const defaultHelper = 'Valor numerico mayor o igual a 0.';
    const categories = [];
    state.featureOrder.forEach((feature) => {
        if (SKIPPED_INPUT_FEATURES.has(feature)) {
            return;
        }
        const details = featureDetails[feature] || {};
        const categoryName = details.category || defaultCategory;
        let categoryEntry = categories.find((item) => item.name === categoryName);
        if (!categoryEntry) {
            categoryEntry = {
                name: categoryName,
                description:
                    featureCategoryDescriptions[categoryName] ||
                    featureCategoryDescriptions[defaultCategory],
                items: [],
            };
            categories.push(categoryEntry);
        }
        categoryEntry.items.push({ feature, details });
    });

    const numericWrapper = document.createElement('div');
    numericWrapper.className = 'feature-section-stack';

    categories.forEach((category) => {
        const section = document.createElement('section');
        section.className = 'feature-section';

        const header = document.createElement('div');
        header.className = 'feature-section-header';

        const title = document.createElement('h3');
        title.textContent = category.name;
        header.appendChild(title);

        if (category.description) {
            const description = document.createElement('p');
            description.textContent = category.description;
            header.appendChild(description);
        }

        section.appendChild(header);

        const grid = document.createElement('div');
        grid.className = 'feature-section-grid';

        category.items.forEach(({ feature, details }) => {
            const inputId = featureInputId(feature);
            const labelText = getFeatureLabel(feature);

            const card = document.createElement('div');
            card.className = 'feature-card';

            const label = document.createElement('label');
            label.setAttribute('for', inputId);
            label.textContent = labelText;

            const input = document.createElement('input');
            input.type = 'number';
            input.step = details.step !== undefined ? String(details.step) : 'any';
            input.id = inputId;
            input.name = feature;
            input.dataset.feature = feature;
            input.dataset.featureLabel = labelText;
            input.inputMode = 'decimal';

            if (details.placeholder) {
                input.placeholder = details.placeholder;
            }

            const constraints = {
                min: details.min !== undefined ? details.min : 0,
                max: details.max,
            };
            if (constraints.max !== undefined && constraints.max !== null) {
                input.max = String(constraints.max);
            }

            enforceNonNegativeInput(input, labelText, constraints);

            const helper = document.createElement('small');
            helper.className = 'feature-helper';
            helper.textContent = details.helper || defaultHelper;

            card.append(label, input, helper);
            grid.appendChild(card);
        });

        section.appendChild(grid);
        numericWrapper.appendChild(section);
    });

    if (numericWrapper.childElementCount > 0) {
        container.appendChild(numericWrapper);
    }
}

function renderGroupedControls(container) {
    groupedFeatureConfigs.forEach((config) => {
        const wrapper = document.createElement('div');
        wrapper.className = 'form-row';

        const label = document.createElement('label');
        label.setAttribute('for', config.selectId);
        label.textContent = config.label;

        const select = document.createElement('select');
        select.id = config.selectId;
        select.required = true;
        select.dataset.groupId = config.id;

        const placeholder = document.createElement('option');
        placeholder.value = '';
        placeholder.textContent = `Selecciona ${config.label.toLowerCase()}`;
        placeholder.disabled = true;
        placeholder.selected = true;
        select.appendChild(placeholder);

        config.options.forEach((option) => {
            const opt = document.createElement('option');
            opt.value = option.value;
            opt.textContent = option.label;
            select.appendChild(opt);
        });

        wrapper.append(label, select);
        container.appendChild(wrapper);
    });
}

function renderSymptomControls(container) {
    if (SYMPTOM_CONFIGS.length === 0) return;
    const card = document.createElement('section');
    card.className = 'symptom-card';

    const heading = document.createElement('h3');
    heading.textContent = 'Sintomas';
    card.appendChild(heading);

    const note = document.createElement('p');
    note.className = 'symptom-note';
    note.textContent = 'Selecciona los sintomas presentes para este caso.';
    card.appendChild(note);

    const grid = document.createElement('div');
    grid.className = 'symptom-grid';

    SYMPTOM_CONFIGS.forEach((item) => {
        const option = document.createElement('label');
        option.className = 'symptom-option';

        const input = document.createElement('input');
        input.type = 'checkbox';
        input.value = item.key;
        input.dataset.symptom = item.key;

        const marker = document.createElement('span');
        marker.className = 'symptom-marker';

        const text = document.createElement('span');
        text.textContent = item.label;

        option.append(input, marker, text);
        grid.appendChild(option);
    });

    card.appendChild(grid);
    container.appendChild(card);
}

function enforceNonNegativeInput(input, labelText = '', constraints = {}) {
    if (!input) return;
    const { min = 0, max = null } = constraints;
    if (min !== null && min !== undefined) {
        input.min = String(min);
    }
    input.addEventListener('input', () => {
        if (input.value === '') {
            input.setCustomValidity('');
            return;
        }
        const value = Number.parseFloat(input.value);
        if (Number.isNaN(value)) {
            input.setCustomValidity(
                labelText ? `Ingresa un numero valido para ${labelText}.` : 'Ingresa un numero valido.'
            );
        } else if (min !== null && min !== undefined && value < min) {
            input.setCustomValidity(
                labelText ? `No se permiten valores negativos en ${labelText}.` : 'No se permiten valores negativos.'
            );
        } else if (max !== null && max !== undefined && value > max) {
            input.setCustomValidity('El dato ingresado es invalido.');
        } else {
            input.setCustomValidity('');
        }
    });
    input.addEventListener('blur', () => {
        if (!input.checkValidity()) {
            input.reportValidity();
        }
    });
}

function readGroupedSelections() {
    const values = {};
    groupedFeatureConfigs.forEach((config) => {
        const select = document.getElementById(config.selectId);
        if (!select) {
            return;
        }
        const selectedValue = select.value;
        if (!selectedValue) {
            throw new Error(`Selecciona una opcion para ${config.label}.`);
        }
        const option = config.options.find((opt) => opt.value === selectedValue);
        if (!option) {
            throw new Error(`Selecciona una opcion para ${config.label}.`);
        }
        config.featureKeys.forEach((key) => {
            values[key] = option.featureValues[key] ?? 0;
        });
    });
    return values;
}

function readSymptomSelections() {
    const values = {};
    SYMPTOM_CONFIGS.forEach((item) => {
        values[item.key] = 0;
    });
    const checkboxes = document.querySelectorAll('input[data-symptom]');
    checkboxes.forEach((checkbox) => {
        const key = checkbox.dataset.symptom;
        if (!key || !SYMPTOM_FEATURE_SET.has(key)) {
            return;
        }
        values[key] = checkbox.checked ? 1 : 0;
    });
    return values;
}

function renderReferenceMetrics() {
    const container = selectors.referenceContent();
    if (!container) return;
    container.innerHTML = '';
    if (state.dataset) {
        const datasetCard = document.createElement('div');
        datasetCard.className = 'metric-card';
        const title = document.createElement('h4');
        title.textContent = 'Dataset operativo';
        const description = document.createElement('strong');
        description.textContent = state.dataset.description || state.dataset.id || 'Dataset';
        const detail = document.createElement('small');
        const target = state.targetColumn || 'diagnosis';
        const labels = state.classLabels?.length
            ? `Clases activas: ${state.classLabels.join(', ')}`
            : 'Clases aun no definidas.';
        detail.textContent = `Objetivo: ${target}. ${labels}`;
        datasetCard.append(title, description, detail);

        const distributionSummary = state.dataset.class_distribution?.summary;
        if (Array.isArray(distributionSummary) && distributionSummary.length > 0) {
            const list = document.createElement('ul');
            list.className = 'distribution-list';
            distributionSummary.forEach((item) => {
                const name = item?.name ?? item?.id ?? 'Clase';
                const original = item?.original ?? 0;
                const balanced = item?.balanced ?? 0;
                const li = document.createElement('li');
                li.textContent = `${name}: ${balanced} balanceado (original ${original})`;
                list.appendChild(li);
            });
            datasetCard.appendChild(list);
        }

        container.appendChild(datasetCard);
    }
    state.models.forEach((model) => {
        const metrics = model.metrics?.metrics || {};
        const block = document.createElement('div');
        block.className = 'metric-card';

        const title = document.createElement('h4');
        title.textContent = model.label;

        const accuracy = document.createElement('strong');
        accuracy.textContent = metrics.accuracy !== undefined
            ? (metrics.accuracy * 100).toFixed(2) + '%'
            : 'N/A';

        const detail = document.createElement('small');
        detail.textContent = metrics.f1 !== undefined
            ? `F1: ${(metrics.f1 * 100).toFixed(1)}%`
            : 'Sin datos de F1.';

        block.append(title, accuracy, detail);
        container.appendChild(block);
    });
}

function updateHeroStats() {
    const classCountEl = selectors.heroClassCount();
    const featureCountEl = selectors.heroFeatureCount();
    if (classCountEl) {
        const classCount = state.classLabels?.length || 0;
        classCountEl.textContent = classCount || '-';
        const hint = classCountEl.nextElementSibling;
        if (hint) {
            hint.textContent = classCount > 0
                ? `Diagnostico: ${state.classLabels.join(', ')}`
                : 'Diagnostico: -';
        }
    }
    if (featureCountEl) {
        const featureCount = state.featureOrder?.length || 0;
        featureCountEl.textContent = featureCount || '-';
    }
}

async function handleIndividualSubmit(event) {
    event.preventDefault();
    const form = event.currentTarget;
    const statusEl = selectors.individualStatus();
    toggleHidden(selectors.individualResult, true);
    setStatus(statusEl, 'Calculando...', '');
    form.querySelector('button[type="submit"]').disabled = true;

    try {
        const model = selectors.individualModel().value;
        const features = {};
        Object.assign(features, readGroupedSelections());
        Object.assign(features, readSymptomSelections());
        const inputs = selectors.featureInputs().querySelectorAll('input[data-feature]');
        inputs.forEach((input) => {
            const feature = input.dataset.feature;
            const labelText = input.dataset.featureLabel || feature;
            const raw = input.value.trim();
            const hasValue = raw !== '';
            const numericValue = hasValue ? Number.parseFloat(raw) : 0;
            if (hasValue && Number.isNaN(numericValue)) {
                throw new Error(`Ingresa un numero valido para ${labelText}.`);
            }
            if (numericValue < 0) {
                throw new Error(`El valor de ${labelText} no puede ser negativo.`);
            }
            features[feature] = numericValue;
            if (hasValue) {
                input.value = raw;
            }
        });

        const response = await fetch('/api/predict/individual', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ model, features }),
        });

        const payload = await response.json();
        if (response.status === 401) {
            throw new Error(payload.error || 'Acceso no autorizado.');
        }
        if (!response.ok) {
            throw new Error(payload.error || 'No se pudo obtener la prediccion.');
        }

        renderIndividualResult(payload);
        setStatus(statusEl, 'Prediccion generada.', 'success');
    } catch (error) {
        console.error(error);
        const message = error instanceof Error ? error.message : 'Error inesperado.';
        setStatus(selectors.individualStatus, message, 'error');
    } finally {
        form.querySelector('button[type="submit"]').disabled = false;
    }
}

function renderIndividualResult(payload) {
    const container = selectors.individualResult();
    if (!container) return;
    container.innerHTML = '';

    if (Array.isArray(payload.class_labels) && payload.class_labels.length) {
        state.classInfo = payload.class_labels;
        state.classLabels = payload.class_labels.map((item) => String(item?.name ?? item?.id ?? item));
        renderReferenceMetrics();
        updateHeroStats();
    }

    const title = document.createElement('h3');
    title.textContent = `Resultado con ${payload.model?.label || payload.model?.id}`;

    const outcome = document.createElement('p');
    const predictedLabel = payload.prediction_label || payload.prediction;
    outcome.innerHTML = `Diagnostico estimado: <strong>${predictedLabel}</strong>`;

    container.append(title, outcome);

    if (Array.isArray(payload.probabilities) && payload.probabilities.length > 0) {
        const list = document.createElement('ul');
        payload.probabilities.forEach((prob, index) => {
            const item = document.createElement('li');
            if (prob && typeof prob === 'object') {
                const labelText = prob.label || state.classLabels[index] || `Clase ${prob.id ?? index}`;
                item.textContent = `${labelText}: ${(prob.probability * 100).toFixed(2)}%`;
            } else {
                const fallback = state.classLabels[index] || `Clase ${index}`;
                item.textContent = `${fallback}: ${(prob * 100).toFixed(2)}%`;
            }
            list.appendChild(item);
        });
        container.appendChild(list);
    }

    toggleHidden(container, false);
}

async function handleBatchSubmit(event) {
    event.preventDefault();
    const form = event.currentTarget;
    const statusEl = selectors.batchStatus();
    setStatus(statusEl, 'Procesando archivo...', '');
    toggleHidden(selectors.batchResult, true);
    form.querySelector('button[type="submit"]').disabled = true;

    try {
        const formData = new FormData(form);
        const response = await fetch('/api/predict/batch', {
            method: 'POST',
            body: formData,
        });
        const payload = await response.json();
        if (response.status === 401) {
            throw new Error(payload.error || 'Acceso no autorizado.');
        }
        if (!response.ok) {
            throw new Error(payload.error || 'No se pudo procesar el archivo.');
        }
        renderBatchResult(payload);
        setStatus(statusEl, `Analizado ${payload.total_samples} registros.`, 'success');
    } catch (error) {
        console.error(error);
        const message = error instanceof Error ? error.message : 'Error inesperado.';
        setStatus(statusEl, message, 'error');
    } finally {
        form.querySelector('button[type="submit"]').disabled = false;
    }
}

function renderBatchResult(payload) {
    if (Array.isArray(payload.class_labels) && payload.class_labels.length) {
        state.classInfo = payload.class_labels;
        state.classLabels = payload.class_labels.map((item) => String(item?.name ?? item?.id ?? item));
    }
    renderBatchMetrics(payload);
    renderConfusionMatrix(payload.confusion_matrix);
    renderBatchPreview(payload.preview);
    updateHeroStats();
    renderReferenceMetrics();
    toggleHidden(selectors.batchResult, false);
}

function renderBatchMetrics(payload) {
    const container = selectors.batchMetrics();
    if (!container) return;
    container.innerHTML = '';
    const metrics = payload.metrics || {};

    const metricLabels = [
        { key: 'accuracy', label: 'Exactitud' },
        { key: 'precision', label: 'Precision' },
        { key: 'recall', label: 'Sensibilidad' },
        { key: 'f1', label: 'F1 armonico' },
    ];

    metricLabels.forEach((item) => {
        const value = metrics[item.key];
        const card = document.createElement('div');
        card.className = 'metric-card';

        const title = document.createElement('h4');
        title.textContent = item.label;

        const strong = document.createElement('strong');
        strong.textContent = value !== undefined ? (value * 100).toFixed(2) + '%' : 'N/A';

        card.append(title, strong);
        container.appendChild(card);
    });
}

function renderConfusionMatrix(matrixPayload) {
    const imageEl = selectors.confusionImage();
    const tableEl = selectors.confusionTable();
    if (!matrixPayload || !imageEl || !tableEl) return;

    const imageSource = matrixPayload.image_png_base64 || matrixPayload.image_counts_png_base64;
    if (imageSource) {
        imageEl.src = `data:image/png;base64,${imageSource}`;
    }

    const normalizedMatrix = Array.isArray(matrixPayload.matrix_normalized)
        ? matrixPayload.matrix_normalized
        : [];
    const countsMatrix = Array.isArray(matrixPayload.matrix) ? matrixPayload.matrix : [];
    const hasCounts = countsMatrix.length > 0;
    const hasNormalized = normalizedMatrix.length > 0;
    const matrix = hasCounts ? countsMatrix : normalizedMatrix;

    const rawLabels = Array.isArray(matrixPayload.labels) ? matrixPayload.labels : [];
    const labels = rawLabels.map((item) => {
        if (item && typeof item === 'object') {
            return item.name ?? item.id ?? '';
        }
        return item ?? '';
    });

    tableEl.innerHTML = '';
    if (matrix.length === 0) return;

    const thead = document.createElement('thead');
    const headRow = document.createElement('tr');
    headRow.appendChild(document.createElement('th'));
    labels.forEach((label) => {
        const th = document.createElement('th');
        th.textContent = `Prediccion ${label}`;
        headRow.appendChild(th);
    });
    thead.appendChild(headRow);

    const tbody = document.createElement('tbody');
    matrix.forEach((row, rowIndex) => {
        const tr = document.createElement('tr');
        const labelCell = document.createElement('th');
        labelCell.textContent = `Valor real ${labels[rowIndex] ?? rowIndex}`;
        tr.appendChild(labelCell);
        row.forEach((value, colIndex) => {
            const td = document.createElement('td');
            const count = hasCounts ? countsMatrix[rowIndex]?.[colIndex] : undefined;
            const ratio = hasNormalized ? normalizedMatrix[rowIndex]?.[colIndex] : undefined;
            if (count !== undefined && ratio !== undefined) {
                const pct = Number.isFinite(ratio) ? (ratio * 100).toFixed(1) : ratio;
                td.textContent = `${count} (${pct}%)`;
            } else if (count !== undefined) {
                td.textContent = count;
            } else if (ratio !== undefined) {
                const pct = Number.isFinite(ratio) ? (ratio * 100).toFixed(1) : ratio;
                td.textContent = `${pct}%`;
            } else {
                td.textContent = value;
            }
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });

    if (hasCounts && hasNormalized) {
        const caption = document.createElement('caption');
        caption.textContent = 'Conteos (porcentaje por fila).';
        tableEl.appendChild(caption);
    }

    tableEl.appendChild(thead);
    tableEl.appendChild(tbody);
}

function renderBatchPreview(rows) {
    const table = selectors.batchPreview();
    if (!table) return;
    table.innerHTML = '';
    if (!Array.isArray(rows) || rows.length === 0) {
        table.textContent = 'Sin registros para mostrar.';
        return;
    }

    const columns = Object.keys(rows[0]);

    const thead = document.createElement('thead');
    const headRow = document.createElement('tr');
    columns.forEach((column) => {
        const th = document.createElement('th');
        th.textContent = column;
        headRow.appendChild(th);
    });
    thead.appendChild(headRow);

    const tbody = document.createElement('tbody');
    rows.forEach((row) => {
        const tr = document.createElement('tr');
        columns.forEach((column) => {
            const td = document.createElement('td');
            const value = row[column];
            td.textContent = typeof value === 'number'
                ? Number.parseFloat(value).toFixed(3)
                : value;
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });

    table.appendChild(thead);
    table.appendChild(tbody);
}

document.addEventListener('DOMContentLoaded', () => {
    const individualForm = selectors.individualForm();
    const batchForm = selectors.batchForm();
    if (individualForm) individualForm.addEventListener('submit', handleIndividualSubmit);
    if (batchForm) batchForm.addEventListener('submit', handleBatchSubmit);
    initTabs();
    loadStatus();
});
function featureInputId(featureName) {
    return `feature-${featureName.replace(/[^a-zA-Z0-9]/g, '_')}`;
}
