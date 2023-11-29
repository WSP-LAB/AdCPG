node_features = {
    'TYPE': {
        'All': {
            'Node_type': {
                'is_grouped': False,
                'types': [
                    'CALL',
                    'IDENTIFIER',
                    'LITERAL',
                    'LOCAL',
                    'METHOD',
                    'METHOD_PARAMETER_IN',
                    'RETURN',
                ],
            },
        },
    },
    'API': {
        'LOCAL': {
            'Identifier': {
                'is_grouped': False,
                'apis': [
                    'Array',
                    'Date',
                    'Document',
                    'History',
                    'IntersectionObserver',
                    'JSON',
                    'Location',
                    'MutationObserver',
                    'Navigator',
                    'RegExp',
                    'Screen',
                    'Storage',
                    'String',
                    'URL',
                    'Window',
                    'XMLHttpRequest',
                ],
            },
        },
        'CALL': {
            'Global': {
                'is_grouped': False,
                'apis': [
                    'decodeURI',
                    'decodeURIComponent',
                    'encodeURI',
                    'encodeURIComponent',
                    'escape',
                    'eval',
                    'unescape',
                ],
            },
            'Object': {
                'is_grouped': True,
                'apis': [
                    'Object',
                    '__defineGetter__',
                    '__defineSetter__',
                    '__lookupGetter__',
                    '__lookupSetter__',
                    '__proto__',
                    'assign',
                    'constructor',
                    'create',
                    'defineProperties',
                    'defineProperty',
                    'freeze',
                    'fromEntries',
                    'getOwnPropertyDescriptor',
                    'getOwnPropertyDescriptors',
                    'getOwnPropertyNames',
                    'getOwnPropertySymbols',
                    'getPrototypeOf',
                    'hasOwnProperty',
                    'is',
                    'isExtensible',
                    'isFrozen',
                    'isPrototypeOf',
                    'isSealed',
                    'preventExtensions',
                    'propertyIsEnumerable',
                    'seal',
                    'setPrototypeOf',
                    'valueOf',
                ],
            },
            'Function': {
                'is_grouped': True,
                'apis': [
                    'Function',
                    'apply',
                    'arguments',
                    'bind',
                    'call',
                    'caller',
                    'displayName',
                    'prototype',
                ],
            },
            'Date': {
                'is_grouped': True,
                'apis': [
                    'Date',
                    'UTC',
                    'getDate',
                    'getDay',
                    'getFullYear',
                    'getHours',
                    'getMilliseconds',
                    'getMinutes',
                    'getMonth',
                    'getSeconds',
                    'getTime',
                    'getTimezoneOffset',
                    'getUTCDate',
                    'getUTCDay',
                    'getUTCFullYear',
                    'getUTCHours',
                    'getUTCMilliseconds',
                    'getUTCMinutes',
                    'getUTCMonth',
                    'getUTCSeconds',
                    'getYear',
                    'now',
                    'parse',
                    'setDate',
                    'setFullYear',
                    'setHours',
                    'setMilliseconds',
                    'setMinutes',
                    'setMonth',
                    'setSeconds',
                    'setTime',
                    'setUTCDate',
                    'setUTCFullYear',
                    'setUTCHours',
                    'setUTCMilliseconds',
                    'setUTCMinutes',
                    'setUTCMonth',
                    'setUTCSeconds',
                    'setYear',
                    'toDateString',
                    'toGMTString',
                    'toISOString',
                    'toLocaleDateString',
                    'toLocaleTimeString',
                    'toTimeString',
                    'toUTCString',
                ],
            },
            'String': {
                'is_grouped': True,
                'apis': [
                    'String',
                    'at',
                    'charAt',
                    'charCodeAt',
                    'codePointAt',
                    'concat',
                    'endsWith',
                    'fromCharCode',
                    'fromCodePoint',
                    'includes',
                    'indexOf',
                    'lastIndexOf',
                    'localeCompare',
                    'match',
                    'matchAll',
                    'normalize',
                    'padEnd',
                    'padStart',
                    'raw',
                    'repeat',
                    'replace',
                    'replaceAll',
                    'search',
                    'slice',
                    'split',
                    'startsWith',
                    'substring',
                    'toLocaleLowerCase',
                    'toLocaleUpperCase',
                    'toLowerCase',
                    'toString',
                    'toUpperCase',
                    'trim',
                    'trimEnd',
                    'trimStart',
                ],
            },
            'RegExp': {
                'is_grouped': True,
                'apis': [
                    'RegExp',
                    'compile',
                    'dotAll',
                    'exec',
                    'flags',
                    'global',
                    'hasIndices',
                    'ignoreCase',
                    'input',
                    'lastIndex',
                    'lastMatch',
                    'lastParen',
                    'leftContext',
                    'multiline',
                    'rightContext',
                    'source',
                    'sticky',
                    'test',
                    'unicode',
                ],
            },
            'Array': {
                'is_grouped': True,
                'apis': [
                    'Array',
                    'at',
                    'concat',
                    'copyWithin',
                    'entries',
                    'every',
                    'fill',
                    'filter',
                    'find',
                    'findIndex',
                    'findLast',
                    'findLastIndex',
                    'flat',
                    'flatMap',
                    'forEach',
                    'from',
                    'group',
                    'groupToMap',
                    'includes',
                    'indexOf',
                    'isArray',
                    'join',
                    'keys',
                    'lastIndexOf',
                    'map',
                    'of',
                    'pop',
                    'push',
                    'reduce',
                    'reduceRight',
                    'reverse',
                    'shift',
                    'slice',
                    'some',
                    'sort',
                    'splice',
                    'unshift',
                    'values',
                ],
            },
            'JSON': {
                'is_grouped': True,
                'apis': [
                    'parse',
                    'stringify',
                    'toJSON',
                ],
            },
            'Event': {
                'is_grouped': True,
                'apis': [
                    'Event',
                    'EventTarget',
                    'addEventListener',
                    'bubbles',
                    'cancelBubble',
                    'cancelable',
                    'composed',
                    'composedPath',
                    'currentTarget',
                    'defaultPrevented',
                    'dispatchEvent',
                    'event',
                    'eventPhase',
                    'explicitOriginalTarget',
                    'initEvent',
                    'isTrusted',
                    'originalTarget',
                    'preventDefault',
                    'removeEventListener',
                    'scoped',
                    'stopImmediatePropagation',
                    'stopPropagation',
                    'target',
                    'timeStamp',
                ],
            },
            'Node': {
                'is_grouped': True,
                'apis': [
                    'appendChild',
                    'baseURI',
                    'childNodes',
                    'cloneNode',
                    'compareDocumentPosition',
                    'contains',
                    'firstChild',
                    'getRootNode',
                    'hasChildNodes',
                    'insertBefore',
                    'isConnected',
                    'isDefaultNamespace',
                    'isEqualNode',
                    'isSameNode',
                    'lastChild',
                    'lookupNamespaceURI',
                    'lookupPrefix',
                    'nextSibling',
                    'nodeName',
                    'nodeType',
                    'nodeValue',
                    'normalize',
                    'ownerDocument',
                    'parentElement',
                    'parentNode',
                    'previousSibling',
                    'removeChild',
                    'replaceChild',
                    'textContent',
                ],
            },
            'Document': {
                'is_grouped': False,
                'apis': [
                    'Document',
                    'activeElement',
                    'adoptNode',
                    'adoptedStyleSheets',
                    'alinkColor',
                    'all',
                    'anchors',
                    'applets',
                    'bgColor',
                    'body',
                    'caretPositionFromPoint',
                    'caretRangeFromPoint',
                    'characterSet',
                    'charset',
                    'compatMode',
                    'contentType',
                    'cookie',
                    'createAttribute',
                    'createAttributeNS',
                    'createCDATASection',
                    'createComment',
                    'createDocumentFragment',
                    'createElement',
                    'createElementNS',
                    'createEntityReference',
                    'createEvent',
                    'createExpression',
                    'createNSResolver',
                    'createNodeIterator',
                    'createProcessingInstruction',
                    'createRange',
                    'createTextNode',
                    'createTouch',
                    'createTouchList',
                    'createTreeWalker',
                    'currentScript',
                    'defaultView',
                    'designMode',
                    'dir',
                    'doctype',
                    'document',
                    'documentElement',
                    'documentURI',
                    'domain',
                    'elementFromPoint',
                    'elementsFromPoint',
                    'embeds',
                    'enableStyleSheetsForSet',
                    'evaluate',
                    'execCommand',
                    'exitFullscreen',
                    'exitPictureInPicture',
                    'exitPointerLock',
                    'featurePolicy',
                    'fgColor',
                    'fonts',
                    'forms',
                    'fragmentDirective',
                    'fullscreen',
                    'fullscreenElement',
                    'fullscreenEnabled',
                    'getElementById',
                    'getElementsByName',
                    'getSelection',
                    'hasFocus',
                    'hasStorageAccess',
                    'head',
                    'hidden',
                    'images',
                    'implementation',
                    'importNode',
                    'inputEncoding',
                    'lastModified',
                    'lastStyleSheetSet',
                    'linkColor',
                    'links',
                    'mozSetImageElement',
                    'pictureInPictureElement',
                    'pictureInPictureEnabled',
                    'pointerLockElement',
                    'preferredStyleSheetSet',
                    'queryCommandEnabled',
                    'queryCommandIndeterm',
                    'queryCommandState',
                    'queryCommandSupported',
                    'queryCommandValue',
                    'readyState',
                    'referrer',
                    'releaseCapture',
                    'requestStorageAccess',
                    'rootElement',
                    'scripts',
                    'scrollingElement',
                    'selectedStyleSheetSet',
                    'styleSheetSets',
                    'styleSheets',
                    'timeline',
                    'title',
                    'visibilityState',
                    'vlinkColor',
                    'xmlEncoding',
                    'xmlStandalone',
                    'xmlVersion',
                ],
            },
            'Element': {
                'is_grouped': False,
                'apis': [
                    'after',
                    'animate',
                    'append',
                    'assignedSlot',
                    'attachShadow',
                    'attributes',
                    'before',
                    'childElementCount',
                    'children',
                    'classList',
                    'className',
                    'clientHeight',
                    'clientLeft',
                    'clientTop',
                    'clientWidth',
                    'closest',
                    'computedStyleMap',
                    'elementTiming',
                    'firstElementChild',
                    'getAnimations',
                    'getAttribute',
                    'getAttributeNS',
                    'getAttributeNames',
                    'getAttributeNode',
                    'getAttributeNodeNS',
                    'getBoundingClientRect',
                    'getBoxQuads',
                    'getClientRects',
                    'getElementsByClassName',
                    'getElementsByTagName',
                    'getElementsByTagNameNS',
                    'hasAttribute',
                    'hasAttributeNS',
                    'hasAttributes',
                    'hasPointerCapture',
                    'innerHTML',
                    'insertAdjacentElement',
                    'insertAdjacentHTML',
                    'insertAdjacentText',
                    'lastElementChild',
                    'localName',
                    'matches',
                    'namespaceURI',
                    'nextElementSibling',
                    'openOrClosedShadowRoot',
                    'outerHTML',
                    'part',
                    'prefix',
                    'prepend',
                    'previousElementSibling',
                    'querySelector',
                    'querySelectorAll',
                    'releasePointerCapture',
                    'remove',
                    'removeAttribute',
                    'removeAttributeNS',
                    'removeAttributeNode',
                    'replaceChildren',
                    'replaceWith',
                    'requestFullscreen',
                    'requestPointerLock',
                    'scroll',
                    'scrollBy',
                    'scrollHeight',
                    'scrollIntoView',
                    'scrollIntoViewIfNeeded',
                    'scrollLeft',
                    'scrollLeftMax',
                    'scrollTo',
                    'scrollTop',
                    'scrollTopMax',
                    'scrollWidth',
                    'setAttribute',
                    'setAttributeNS',
                    'setAttributeNode',
                    'setAttributeNodeNS',
                    'setCapture',
                    'setHTML',
                    'setPointerCapture',
                    'shadowRoot',
                    'slot',
                    'tagName',
                    'toggleAttribute',
                ],
            },
            'Window': {
                'is_grouped': False,
                'apis': [
                    'alert',
                    'atob',
                    'blur',
                    'btoa',
                    'caches',
                    'cancelAnimationFrame',
                    'cancelIdleCallback',
                    'captureEvents',
                    'clearImmediate',
                    'clearInterval',
                    'clearTimeout',
                    'clientInformation',
                    'closed',
                    'confirm',
                    'console',
                    'content',
                    'createImageBitmap',
                    'credentialless',
                    'crypto',
                    'customElements',
                    'defaultStatus',
                    'devicePixelRatio',
                    'dump',
                    'external',
                    'fetch',
                    'focus',
                    'frameElement',
                    'frames',
                    'fullScreen',
                    'getComputedStyle',
                    'getDefaultComputedStyle',
                    'getSelection',
                    'indexedDB',
                    'innerHeight',
                    'innerWidth',
                    'isSecureContext',
                    'locationbar',
                    'matchMedia',
                    'menubar',
                    'messageManager',
                    'moveBy',
                    'moveTo',
                    'navigation',
                    'opener',
                    'outerHeight',
                    'outerWidth',
                    'pageXOffset',
                    'pageYOffset',
                    'parent',
                    'personalbar',
                    'postMessage',
                    'print',
                    'prompt',
                    'queryLocalFonts',
                    'releaseEvents',
                    'reportError',
                    'requestAnimationFrame',
                    'requestIdleCallback',
                    'resizeBy',
                    'resizeTo',
                    'scheduler',
                    'scrollByLines',
                    'scrollByPages',
                    'scrollMaxX',
                    'scrollMaxY',
                    'scrollX',
                    'scrollY',
                    'scrollbars',
                    'self',
                    'setImmediate',
                    'setInterval',
                    'setResizable',
                    'setTimeout',
                    'showDirectoryPicker',
                    'showModalDialog',
                    'showOpenFilePicker',
                    'showSaveFilePicker',
                    'sidebar',
                    'sizeToContent',
                    'speechSynthesis',
                    'status',
                    'statusbar',
                    'stop',
                    'toolbar',
                    'top',
                    'updateCommands',
                    'visualViewport',
                    'window',
                ],
            },
            'Navigator': {
                'is_grouped': True,
                'apis': [
                    'activeVRDisplays',
                    'appCodeName',
                    'appName',
                    'appVersion',
                    'buildID',
                    'canShare',
                    'clearAppBadge',
                    'connection',
                    'contacts',
                    'cookieEnabled',
                    'credentials',
                    'deviceMemory',
                    'doNotTrack',
                    'geolocation',
                    'getBattery',
                    'getUserMedia',
                    'getVRDisplays',
                    'globalPrivacyControl',
                    'hardwareConcurrency',
                    'hid',
                    'ink',
                    'javaEnabled',
                    'keyboard',
                    'language',
                    'languages',
                    'locks',
                    'maxTouchPoints',
                    'mediaCapabilities',
                    'mediaDevices',
                    'mediaSession',
                    'mimeTypes',
                    'navigator',
                    'onLine',
                    'oscpu',
                    'pdfViewerEnabled',
                    'permissions',
                    'platform',
                    'plugins',
                    'presentation',
                    'product',
                    'productSub',
                    'registerProtocolHandler',
                    'requestMIDIAccess',
                    'requestMediaKeySystemAccess',
                    'securitypolicy',
                    'sendBeacon',
                    'serial',
                    'serviceWorker',
                    'setAppBadge',
                    'share',
                    'standalone',
                    'storage',
                    'taintEnabled',
                    'unregisterProtocolHandler',
                    'userActivation',
                    'userAgent',
                    'userAgentData',
                    'vendor',
                    'vendorSub',
                    'vibrate',
                    'virtualKeyboard',
                    'wakeLock',
                    'webdriver',
                    'windowControlsOverlay',
                    'xr',
                ],
            },
            'Screen': {
                'is_grouped': True,
                'apis': [
                    'availHeight',
                    'availLeft',
                    'availTop',
                    'availWidth',
                    'colorDepth',
                    'height',
                    'left',
                    'lockOrientation',
                    'mozBrightness',
                    'mozEnabled',
                    'orientation',
                    'pixelDepth',
                    'screen',
                    'screenLeft',
                    'screenTop',
                    'screenX',
                    'screenY',
                    'top',
                    'unlockOrientation',
                    'width',
                ],
            },
            'Storage': {
                'is_grouped': True,
                'apis': [
                    'getItem',
                    'key',
                    'localStorage',
                    'removeItem',
                    'sessionStorage',
                    'setItem',
                ],
            },
            'Location': {
                'is_grouped': True,
                'apis': [
                    'ancestorOrigins',
                    'assign',
                    'hash',
                    'host',
                    'hostname',
                    'href',
                    'location',
                    'origin',
                    'pathname',
                    'port',
                    'protocol',
                    'reload',
                    'replace',
                    'search',
                ],
            },
            'History': {
                'is_grouped': True,
                'apis': [
                    'back',
                    'forward',
                    'go',
                    'history',
                    'pushState',
                    'replaceState',
                    'scrollRestoration',
                    'state',
                ],
            },
            'URL': {
                'is_grouped': True,
                'apis': [
                    'URL',
                    'createObjectURL',
                    'hash',
                    'host',
                    'hostname',
                    'href',
                    'origin',
                    'password',
                    'pathname',
                    'port',
                    'protocol',
                    'revokeObjectURL',
                    'search',
                    'searchParams',
                    'username',
                ],
            },
            'Request': {
                'is_grouped': True,
                'apis': [
                    'XMLHttpRequest',
                    'abort',
                    'channel',
                    'getAllResponseHeaders',
                    'getResponseHeader',
                    'mozAnon',
                    'mozBackgroundRequest',
                    'mozSystem',
                    'overrideMimeType',
                    'readyState',
                    'response',
                    'responseText',
                    'responseType',
                    'responseURL',
                    'responseXML',
                    'send',
                    'setRequestHeader',
                    'status',
                    'statusText',
                    'timeout',
                    'upload',
                    'withCredentials',
                ],
            },
            'Performance': {
                'is_grouped': True,
                'apis': [
                    'clearMarks',
                    'clearMeasures',
                    'clearResourceTimings',
                    'connectEnd',
                    'connectStart',
                    'domComplete',
                    'domContentLoadedEventEnd',
                    'domContentLoadedEventStart',
                    'domInteractive',
                    'domLoading',
                    'domainLookupEnd',
                    'domainLookupStart',
                    'eventCounts',
                    'fetchStart',
                    'getEntries',
                    'getEntriesByName',
                    'getEntriesByType',
                    'loadEventEnd',
                    'loadEventStart',
                    'mark',
                    'measure',
                    'measureUserAgentSpecificMemory',
                    'memory',
                    'navigation',
                    'navigationStart',
                    'performance',
                    'redirectEnd',
                    'redirectStart',
                    'requestStart',
                    'responseEnd',
                    'responseStart',
                    'secureConnectionStart',
                    'setResourceTimingBufferSize',
                    'timeOrigin',
                    'timing',
                    'unloadEventEnd',
                    'unloadEventStart',
                ],
            },
            'Observer': {
                'is_grouped': True,
                'apis': [
                    'IntersectionObserver',
                    'MutationObserver',
                    'boundingClientRect',
                    'disconnect',
                    'disconnect',
                    'intersectionRatio',
                    'intersectionRect',
                    'isIntersecting',
                    'observe',
                    'observe',
                    'root',
                    'rootBounds',
                    'rootMargin',
                    'takeRecords',
                    'takeRecords',
                    'thresholds',
                    'unobserve',
                ],
            },
            'HTMLElement': {
                'is_grouped': True,
                'apis': [
                    'accessKey',
                    'accessKeyLabel',
                    'align',
                    'attachInternals',
                    'attributeStyleMap',
                    'click',
                    'contentEditable',
                    'crossOrigin',
                    'dataset',
                    'draggable',
                    'enterKeyHint',
                    'fetchPriority',
                    'inert',
                    'innerText',
                    'inputMode',
                    'isContentEditable',
                    'lang',
                    'longDesc',
                    'noModule',
                    'nonce',
                    'offsetHeight',
                    'offsetLeft',
                    'offsetParent',
                    'offsetTop',
                    'offsetWidth',
                    'outerText',
                    'properties',
                    'referrerPolicy',
                    'spellcheck',
                    'src',
                    'style',
                    'tabIndex',
                    'translate',
                ],
            },
            'HTMLScriptElement': {
                'is_grouped': True,
                'apis': [
                    'async',
                    'defer',
                    'supports',
                    'text',
                ],
            },
            'HTMLIFrameElement': {
                'is_grouped': True,
                'apis': [
                    'allow',
                    'allowPaymentRequest',
                    'allowfullscreen',
                    'contentDocument',
                    'contentWindow',
                    'csp',
                    'frameBorder',
                    'marginHeight',
                    'marginWidth',
                    'sandbox',
                    'scrolling',
                    'srcdoc',
                ],
            },
            'HTMLStyleElement': {
                'is_grouped': True,
                'apis': [
                    'disabled',
                    'media',
                    'sheet',
                ],
            },
            'HTMLImageElement': {
                'is_grouped': True,
                'apis': [
                    'Image',
                    'alt',
                    'border',
                    'complete',
                    'currentSrc',
                    'decode',
                    'decoding',
                    'hspace',
                    'isMap',
                    'loading',
                    'naturalHeight',
                    'naturalWidth',
                    'sizes',
                    'srcset',
                    'useMap',
                    'vspace',
                ],
            },
            'HTMLMediaElement': {
                'is_grouped': True,
                'apis': [
                    'Audio',
                    'addTextTrack',
                    'audioTracks',
                    'autoPictureInPicture',
                    'autoplay',
                    'buffered',
                    'canPlayType',
                    'captureStream',
                    'controller',
                    'controls',
                    'controlsList',
                    'currentSrc',
                    'currentTime',
                    'defaultMuted',
                    'defaultPlaybackRate',
                    'disablePictureInPicture',
                    'disableRemotePlayback',
                    'duration',
                    'ended',
                    'error',
                    'fastSeek',
                    'getVideoPlaybackQuality',
                    'load',
                    'loop',
                    'mediaGroup',
                    'mediaKeys',
                    'mozAudioCaptured',
                    'mozCaptureStream',
                    'mozCaptureStreamUntilEnded',
                    'mozFragmentEnd',
                    'mozGetMetadata',
                    'muted',
                    'networkState',
                    'pause',
                    'paused',
                    'play',
                    'playbackRate',
                    'played',
                    'poster',
                    'preload',
                    'preservesPitch',
                    'requestPictureInPicture',
                    'seekToNextFrame',
                    'seekable',
                    'seeking',
                    'setMediaKeys',
                    'setSinkId',
                    'sinkId',
                    'srcObject',
                    'textTracks',
                    'videoHeight',
                    'videoTracks',
                    'videoWidth',
                    'volume',
                ],
            },
            'HTMLCanvasElement': {
                'is_grouped': True,
                'apis': [
                    'captureStream',
                    'getContext',
                    'mozOpaque',
                    'mozPrintCallback',
                    'toBlob',
                    'toDataURL',
                    'transferControlToOffscreen',
                ],
            },
        },
    },
    'STR': {
        'LITERAL': {
            'Tag': {
                'is_grouped': True,
                'strings': [
                    'audio',
                    'canvas',
                    'iframe',
                    'img',
                    'link',
                    'script',
                    'source',
                    'style',
                    'video',
                ],
            },
            'Cookie': {
                'is_grouped': True,
                'strings': [
                    'domain',
                    'expires',
                    'httponly',
                    'max-age',
                    'partitioned',
                    'path',
                    'samesite',
                    'secure',
                ],
            },
            'Event': {
                'is_grouped': True,
                'strings': [
                    'DOMContentLoaded',
                    'DOMNodeInserted',
                    'DOMNodeInsertedIntoDocument',
                    'DOMNodeRemoved',
                    'DOMNodeRemovedFromDocument',
                    'DOMSubtreeModified',
                    'animationend',
                    'animationiteration',
                    'animationstart',
                    'beforeinput',
                    'beforeunload',
                    'blur',
                    'change',
                    'click',
                    'dblclick',
                    'error',
                    'focus',
                    'focusin',
                    'focusout',
                    'hashchange',
                    'input',
                    'keydown',
                    'keypress',
                    'keyup',
                    'load',
                    'mousedown',
                    'mousemove',
                    'mouseover',
                    'mouseup',
                    'resize',
                    'scroll',
                    'submit',
                    'transitionend',
                    'unload',
                ],
            },
            'Resource': {
                'is_grouped': True,
                'strings': [
                    'beacon',
                    'csp_report',
                    'font',
                    'image',
                    'imageset',
                    'main_frame',
                    'media',
                    'object',
                    'object_subrequest',
                    'other',
                    'ping',
                    'script',
                    'speculative',
                    'stylesheet',
                    'sub_frame',
                    'web_manifest',
                    'websocket',
                    'xml_dtd',
                    'xmlhttprequest',
                    'xslt',
                ]
            },
        },
    },
}
edge_features = {
    'TYPE': {
        'All': {
            'Edge_type': {
                'is_grouped': False,
                'types': [
                    'AST',
                    'CFG',
                    'PDG',
                ],
            },
        },
    },
}
