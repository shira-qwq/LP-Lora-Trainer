<script setup lang="ts">
/**
 * Training tasks: queue + detail (metrics, loss curve, log tail, control).
 * Unparseable metrics show as 未知; stop vs. force-kill are distinct actions,
 * and only the recovery operations the adapter declares are offered.
 */
import { computed, nextTick, onMounted, onBeforeUnmount, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import AppIcon from '@/components/AppIcon.vue'
import StatusTag from '@/components/StatusTag.vue'
import ProgressBar from '@/components/ProgressBar.vue'
import LossChart from '@/components/LossChart.vue'
import EmptyState from '@/components/EmptyState.vue'
import ImageViewer, { type ViewerItem } from '@/components/ImageViewer.vue'
import ModalDialog from '@/components/ModalDialog.vue'
import SegTabs from '@/components/SegTabs.vue'
import { api } from '@/api'
import { revealPath } from '@/api/bridge'
import TrainingExportDialog from '@/components/TrainingExportDialog.vue'
import { snapshotTrainingExport } from '@/features/trainingExport'
import { removeTask, store } from '@/features/store'
import { toast, toastError } from '@/features/toast'
import { useArtifacts } from '@/features/useArtifacts'
import { useTaskLog, useTaskMetrics } from '@/features/useTask'
import { groupSamples } from '@/features/samples'
import {
  ACTIVE_TASK_STATES,
  engineName,
  fmtDuration,
  fmtLoss,
  fmtNum,
  fmtTime,
  progressRatio,
  taskStateMeta,
} from '@/features/format'
import type { TaskRuntime, TaskSummary, TrainingExportSource } from '@/types/api'

const route = useRoute()
const router = useRouter()

type Filter = 'all' | 'active' | 'done' | 'failed'
const filter = ref<Filter>('all')
const list = computed(() => {
  const t = store.tasks
  if (filter.value === 'active') return t.filter((x) => ACTIVE_TASK_STATES.includes(x.state))
  if (filter.value === 'done') return t.filter((x) => ['succeeded', 'stopped'].includes(x.state))
  if (filter.value === 'failed') return t.filter((x) => ['failed', 'connection_lost'].includes(x.state))
  return t
})

const selectedId = computed<string | null>(() => {
  const p = route.params.taskId
  const id = typeof p === 'string' && p ? p : null
  return (id && store.tasks.some(t => t.task_id === id) ? id : null) ?? store.tasks[0]?.task_id ?? null
})
const task = computed(() => store.tasks.find((t) => t.task_id === selectedId.value) ?? null)

function select(t: TaskSummary) {
  void router.replace({ name: 'tasks', params: { taskId: t.task_id } })
}

/* A menu owns its target; changing selection must never redirect an action. */
const menu = ref<{ target: TaskSummary; x: number; y: number } | null>(null)
const menuEl = ref<HTMLElement | null>(null)
let menuAnchor: HTMLElement | null = null
const deleteTarget = ref<TaskSummary | null>(null)
const historyBusy = ref(false)
const exportSource = ref<TrainingExportSource | null>(null)
const deletable = (t: TaskSummary) => ['succeeded', 'failed', 'stopped'].includes(t.state)
function closeMenu() { menu.value = null }
async function openMenu(t: TaskSummary, event: MouseEvent | KeyboardEvent) {
  event.preventDefault()
  menuAnchor = event.currentTarget as HTMLElement
  const rect = menuAnchor.getBoundingClientRect()
  const x = event instanceof MouseEvent ? event.clientX : rect.left
  const y = event instanceof MouseEvent ? event.clientY : rect.bottom
  menu.value = { target: t, x, y }
  await nextTick()
  const bounds = menuEl.value?.getBoundingClientRect()
  if (menu.value && bounds) {
    menu.value.x = Math.max(8, Math.min(x, window.innerWidth - bounds.width - 8))
    menu.value.y = Math.max(8, Math.min(y, window.innerHeight - bounds.height - 8))
    menuEl.value?.querySelector<HTMLButtonElement>('button:not(:disabled)')?.focus()
  }
}
function outsideMenu(event: PointerEvent) {
  if (!menuEl.value?.contains(event.target as Node)) closeMenu()
}
function menuScroll(event: Event) {
  // Live log autoscroll is unrelated to the menu's anchor.
  const target = event.target
  if (target instanceof Node && menuAnchor && target.contains(menuAnchor)) closeMenu()
}
function menuKey(event: KeyboardEvent) {
  if (!menu.value) return
  if (event.key === 'Escape') { event.preventDefault(); closeMenu(); menuAnchor?.focus(); return }
  if (event.key === 'Tab') { closeMenu(); return }
  if (!['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) return
  event.preventDefault()
  const buttons = Array.from(menuEl.value?.querySelectorAll<HTMLButtonElement>('button:not(:disabled)') ?? [])
  const index = buttons.indexOf(document.activeElement as HTMLButtonElement)
  const next = event.key === 'Home' ? 0 : event.key === 'End' ? buttons.length - 1 :
    (index + (event.key === 'ArrowDown' ? 1 : -1) + buttons.length) % buttons.length
  buttons[next]?.focus()
}
async function historyAction(kind: 'export' | 'use' | 'delete', target: TaskSummary) {
  closeMenu()
  if (historyBusy.value) return
  if (kind === 'delete') { if (deletable(target)) deleteTarget.value = target; return }
  historyBusy.value = true
  try {
    if (kind === 'use') {
      await router.push({ name: 'train', query: { fromTask: target.task_id } })
    } else {
      const config = await api.tasks.configuration(target.task_id)
      // History exported before overrides were tracked has no list; its stored params
      // are exactly what the old flow wrote in full, so export them as overrides.
      exportSource.value = snapshotTrainingExport({
        name: config.name, output_name: config.output_name, installation_id: config.installation_id,
        architecture: config.architecture, base_model_path: config.base_model_path,
        dataset_id: config.dataset_id, output_dir: config.output_dir, params: { ...config.params },
        explicit_params: Array.isArray(config.explicit_params)
          ? [...config.explicit_params]
          : Object.keys(config.params),
      }, config.engine_id, config.dataset_path)
    }
  } catch (err) { toastError(err, kind === 'use' ? '使用配置' : '导出配置') }
  finally { historyBusy.value = false }
}
async function deleteTask() {
  const target = deleteTarget.value
  if (!target || historyBusy.value) return
  historyBusy.value = true
  try {
    await api.tasks.delete(target.task_id)
    removeTask(target.task_id)
    deleteTarget.value = null
    if (route.params.taskId === target.task_id) await router.replace({ name: 'tasks' })
    toast('已删除任务记录及日志，训练产物保留', 'ok')
  } catch (err) { toastError(err, '删除任务') }
  finally { historyBusy.value = false }
}
onMounted(() => {
  window.addEventListener('pointerdown', outsideMenu)
  window.addEventListener('keydown', menuKey)
  window.addEventListener('resize', closeMenu)
  document.addEventListener('scroll', menuScroll, true)
})
onBeforeUnmount(() => {
  window.removeEventListener('pointerdown', outsideMenu)
  window.removeEventListener('keydown', menuKey)
  window.removeEventListener('resize', closeMenu)
  document.removeEventListener('scroll', menuScroll, true)
})

const { points } = useTaskMetrics(selectedId)
const logLimit = ref(500)
api.settings
  .get()
  .then((s) => (logLimit.value = s.log_tail_lines))
  .catch(() => {})
const { lines } = useTaskLog(selectedId, logLimit)

/* ---- Latest samples, next to the loss curve (same screen) ---- */
const { artifacts } = useArtifacts(selectedId)
const sampleGroups = computed(() => groupSamples(artifacts.value))
/** null = follow the newest step; a key pins the history view until toggled again. */
const pinnedStep = ref<string | null>(null)
const activeGroup = computed(() => {
  const groups = sampleGroups.value
  if (!groups.length) return null
  return (pinnedStep.value && groups.find((g) => g.key === pinnedStep.value)) || groups[groups.length - 1]
})
const activeImages = computed(() => activeGroup.value?.items ?? [])
const heroImage = computed(() => activeImages.value[0] ?? null)
const viewerItems = computed<ViewerItem[]>(() => activeImages.value.map((a) => ({
  src: a.preview_url ? `${a.preview_url}?full=true` : null,
  title: a.file_name,
  subtitle: activeGroup.value?.label,
})))
const viewerIndex = ref<number | null>(null)
function pinStep(key: string) {
  pinnedStep.value = activeGroup.value?.key === key ? null : key
}

/* ---- Run facts: what the engine reported, the live GPU and the submitted config ---- */
const runtime = ref<TaskRuntime | null>(null)
const submitted = ref<Record<string, string | number | boolean | null>>({})
watch(selectedId, async (id) => {
  runtime.value = null
  submitted.value = {}
  if (!id) return
  const [rt, cfg] = await Promise.allSettled([api.tasks.runtime(id), api.tasks.configuration(id)])
  if (id !== selectedId.value) return
  if (rt.status === 'fulfilled') runtime.value = rt.value
  if (cfg.status === 'fulfilled') submitted.value = cfg.value.params ?? {}
}, { immediate: true })

const gpu = computed(() => store.system?.gpu ?? null)
const profileText = computed(() => {
  const p = runtime.value?.memory_profile ?? {}
  if (!p.resolved) return null
  return p.requested && p.requested !== p.resolved ? `${p.resolved}（${p.requested} 自动解析）` : p.resolved
})
const placementText = computed(() => {
  const p = runtime.value?.placement ?? {}
  const parts = []
  if (p.text_encoder) parts.push(`文本 ${p.text_encoder}`)
  if (p.vae) parts.push(`VAE ${p.vae}`)
  return parts.length ? parts.join(' · ') : null
})
const precisionText = computed(() => {
  const r = runtime.value?.run ?? {}
  const parts = [r.precision, r.algo].filter(Boolean)
  return parts.length ? parts.join(' · ') : null
})
const peakText = computed(() => {
  const v = runtime.value?.vram ?? {}
  const peak = v.peak_reserved_gib ?? v.train_peak_reserved_gib
  return peak === undefined ? null : `${peak.toFixed(2)} GiB`
})
const peakHint = computed(() => {
  const v = runtime.value?.vram ?? {}
  if (v.peak_allocated_gib !== undefined) {
    return `引擎自报峰值：allocated ${v.peak_allocated_gib.toFixed(2)} GiB / reserved ${v.peak_reserved_gib?.toFixed(2) ?? '—'} GiB`
  }
  if (v.train_peak_reserved_gib !== undefined) {
    return `引擎自报训练峰值：step ${v.train_peak_step ?? '—'}，reserved ${v.train_peak_reserved_gib.toFixed(2)} GiB`
  }
  return '引擎日志未记录显存峰值'
})
const tokensText = computed(() => {
  const value = submitted.value.MAX_TOKENS
  return value === null || value === undefined || value === '' ? null : String(value)
})

/* Sampling prompts are fixed with the run: they come from the submitted configuration. */
function parseList(text: unknown): string[] {
  if (typeof text !== 'string' || !text.trim()) return []
  try {
    const value = JSON.parse(text)
    return Array.isArray(value) ? value.map((item) => String(item)) : []
  } catch {
    return []
  }
}
const prompts = computed(() => parseList(submitted.value.SAMPLE_PROMPTS))
const seeds = computed(() => parseList(submitted.value.SAMPLE_SEEDS))
const sampleSettings = computed(() => {
  const v = submitted.value
  const parts = []
  if (v.SAMPLE_STEPS) parts.push(`${v.SAMPLE_STEPS} steps`)
  if (v.SAMPLE_CFG !== undefined && v.SAMPLE_CFG !== '') parts.push(`cfg ${v.SAMPLE_CFG}`)
  if (v.SAMPLE_SHIFT !== undefined && v.SAMPLE_SHIFT !== '') parts.push(`shift ${v.SAMPLE_SHIFT}`)
  if (typeof v.SAMPLE_SIZE === 'string' && v.SAMPLE_SIZE.trim()) parts.push(v.SAMPLE_SIZE.trim())
  if (v.SAMPLE_EVERY) parts.push(`每 ${v.SAMPLE_EVERY} 步`)
  return parts.join(' · ')
})
async function copyText(text: string, what: string) {
  try {
    await navigator.clipboard.writeText(text)
    toast(`已复制${what}`, 'ok')
  } catch (err) {
    toastError(err, '复制')
  }
}

/* Log view */
const logQuery = ref('')
const follow = ref(true)
const logEl = ref<HTMLElement>()
const shownLines = computed(() => {
  const q = logQuery.value.trim().toLowerCase()
  return q ? lines.value.filter((l) => l.toLowerCase().includes(q)) : lines.value
})
watch(
  () => shownLines.value.length,
  async () => {
    if (!follow.value) return
    await nextTick()
    logEl.value?.scrollTo({ top: logEl.value.scrollHeight })
  },
)
function onLogScroll() {
  const el = logEl.value
  if (!el) return
  follow.value = el.scrollHeight - el.scrollTop - el.clientHeight < 24
}
function lineClass(l: string) {
  if (/error|traceback|exception/i.test(l)) return 'is-err'
  if (/warn/i.test(l)) return 'is-warn'
  if (l.startsWith('[supervisor]')) return 'is-sys'
  return ''
}

const chartTable = ref(false)

/* Control */
const confirmKind = ref<'stop' | 'kill' | null>(null)
const acting = ref(false)
const canStop = computed(() => !!task.value && ['queued', 'preparing', 'running', 'connection_lost'].includes(task.value.state))
const canKill = computed(() => !!task.value && ['preparing', 'running', 'stopping'].includes(task.value.state))

async function doStop() {
  const t = task.value
  const force = confirmKind.value === 'kill'
  if (!t) return
  acting.value = true
  try {
    if (t.state === 'connection_lost') {
      await api.tasks.acknowledgeExit(t.task_id)
      toast('已记录人工确认，队列可继续执行', 'ok')
    } else {
      await api.tasks.stop(t.task_id, force)
      toast(force ? '已请求强制结束进程树' : '已请求中断，不保证保存 checkpoint', 'ok')
    }
    confirmKind.value = null
  } catch (err) {
    toastError(err, force ? '强制结束' : '停止')
  } finally {
    acting.value = false
  }
}

async function reveal() {
  if (!task.value) return
  const ok = await revealPath(task.value.output_dir)
  if (!ok) toast('仅在桌面窗口中可打开文件夹', 'info')
}

const ratio = computed(() => (task.value ? progressRatio(task.value.progress.step, task.value.progress.total_steps) : null))
const duration = computed(() => {
  const t = task.value
  if (!t?.started_at) return null
  const end = t.finished_at ? new Date(t.finished_at).getTime() : Date.now()
  return (end - new Date(t.started_at).getTime()) / 1000
})
</script>

<template>
  <div class="tasks">
    <!-- Queue -->
    <aside class="tasks__list panel">
      <div class="list__head">
        <h1 class="title-md">训练任务</h1>
        <span class="spacer" />
        <RouterLink :to="{ name: 'train' }" class="btn btn--sm btn--primary">
          <AppIcon name="plus" :size="14" />
          新建
        </RouterLink>
      </div>
      <div class="list__filter">
        <SegTabs
          v-model="filter"
          :options="[
            { value: 'all', label: '全部' },
            { value: 'active', label: '进行中' },
            { value: 'done', label: '已结束' },
            { value: 'failed', label: '失败' },
          ]"
        />
      </div>
      <div class="list__body scroll">
        <button
          v-for="t in list"
          :key="t.task_id"
          class="titem"
          :class="{ 'is-on': t.task_id === selectedId }"
          @click="select(t)"
          @contextmenu="openMenu(t, $event)"
          @keydown.shift.f10="openMenu(t, $event)"
          @keydown="($event.key === 'ContextMenu') && openMenu(t, $event)"
        >
          <span class="row">
            <strong class="truncate">{{ t.name }}</strong>
            <span class="spacer" />
            <StatusTag v-bind="taskStateMeta[t.state]" />
          </span>
          <span class="titem__meta">{{ engineName(t.engine_id) }} · {{ t.architecture }} · {{ fmtTime(t.created_at) }}</span>
          <ProgressBar v-if="t.state === 'running'" :ratio="progressRatio(t.progress.step, t.progress.total_steps)" :light="t.task_id === selectedId" />
        </button>
        <p v-if="store.tasksLoaded && !list.length" class="muted list__empty">
          {{ store.loadError ? `加载失败：${store.loadError}` : '没有任务' }}
        </p>
      </div>
    </aside>

    <!-- Detail -->
    <section v-if="task" :key="task.task_id" class="tasks__detail">
      <div class="detail__top panel enter">
        <header class="detail__head">
          <div class="stack" style="gap: 4px; min-width: 0">
            <span class="eyebrow">{{ task.task_id }} · {{ engineName(task.engine_id) }} / {{ task.installation_label }} · {{ task.architecture }}</span>
            <h2 class="title-lg truncate selectable">{{ task.name }}</h2>
          </div>
          <span class="spacer" />
          <StatusTag v-bind="taskStateMeta[task.state]" />
          <button class="btn btn--ghost btn--sm" :disabled="historyBusy" aria-haspopup="menu" @click="openMenu(task, $event)">更多操作</button>
          <button class="btn btn--ghost btn--sm" :disabled="!canStop || acting" @click="confirmKind = 'stop'">
            <AppIcon name="stop" :size="13" />
            {{ task.state === 'connection_lost' ? '确认已退出' : task.state === 'queued' ? '取消排队' : '中断' }}
          </button>
          <button class="btn btn--danger btn--sm" :disabled="!canKill" @click="confirmKind = 'kill'">强制结束</button>
        </header>

        <div v-if="task.error_summary" class="err selectable">
          <AppIcon name="alert" :size="15" />
          <span class="mono">{{ task.error_summary }}</span>
        </div>

        <div class="metrics">
          <div class="m">
            <span class="m__l">Step</span>
            <span class="m__v">{{ fmtNum(task.progress.step) }}<small> / {{ fmtNum(task.progress.total_steps) }}</small></span>
          </div>
          <div class="m">
            <span class="m__l">Epoch</span>
            <span class="m__v">{{ fmtNum(task.progress.epoch) }}</span>
          </div>
          <div class="m">
            <span class="m__l">Loss</span>
            <span class="m__v">{{ fmtLoss(task.progress.loss) }}</span>
          </div>
          <div class="m">
            <span class="m__l">速度</span>
            <span class="m__v">{{ fmtNum(task.progress.it_per_sec, 2) }}<small v-if="task.progress.it_per_sec !== null"> it/s</small></span>
          </div>
          <div class="m">
            <span class="m__l">已用时</span>
            <span class="m__v">{{ fmtDuration(duration) }}</span>
          </div>
          <div class="m">
            <span class="m__l">预计剩余</span>
            <span class="m__v">{{ ACTIVE_TASK_STATES.includes(task.state) ? fmtDuration(task.progress.eta_seconds) : '—' }}</span>
          </div>
        </div>
        <ProgressBar :ratio="ratio" :segments="24" />

        <!-- What is really in effect: engine-reported facts, live monitor, submitted config. -->
        <div class="runtime">
          <div class="rt" :title="gpu ? `实时监控（nvidia-smi）：${gpu.name}` : '后端未提供 GPU 监控'">
            <span class="rt__l">GPU</span>
            <span class="rt__v">
              {{ gpu ? gpu.name.replace('NVIDIA GeForce ', '') : '—' }}
              <small v-if="gpu">{{ (gpu.memory_used_mb / 1024).toFixed(1) }} / {{ (gpu.memory_total_mb / 1024).toFixed(1) }} GB · {{ gpu.utilization ?? '—' }}%</small>
              <small v-else>未检测到</small>
            </span>
          </div>
          <div class="rt" title="引擎日志：MEMORY_PROFILE 的解析结果（不是 LP 的界面默认值）">
            <span class="rt__l">生效显存档位</span>
            <span class="rt__v">
              {{ profileText ?? '—' }}
              <small>{{ profileText ? '引擎日志' : '日志未记录' }}</small>
            </span>
          </div>
          <div class="rt" title="引擎日志：文本编码器 / VAE 的实际放置方式">
            <span class="rt__l">放置</span>
            <span class="rt__v">
              {{ placementText ?? '—' }}
              <small>{{ placementText ? '引擎日志' : '日志未记录' }}</small>
            </span>
          </div>
          <div class="rt" title="引擎日志：本次运行的算法与精度">
            <span class="rt__l">精度</span>
            <span class="rt__v">
              {{ precisionText ?? '—' }}
              <small>{{ runtime?.run?.device ?? '—' }}</small>
            </span>
          </div>
          <div class="rt" :title="peakHint">
            <span class="rt__l">显存峰值（引擎）</span>
            <span class="rt__v">
              {{ peakText ?? '—' }}
              <small>{{ peakText ? 'reserved' : '日志未记录' }}</small>
            </span>
          </div>
          <div class="rt" title="提交配置：MAX_TOKENS 是每批 token 预算，不是固定 batch size">
            <span class="rt__l">Token 预算</span>
            <span class="rt__v">
              {{ tokensText ?? '—' }}
              <small>{{ tokensText ? 'MAX_TOKENS / 批' : '提交配置缺失' }}</small>
            </span>
          </div>
        </div>

        <div class="detail__grid">
          <div class="chart-wrap">
            <div class="row chart-head">
              <span class="label-pill">Loss</span>
              <span class="muted num">{{ points.length }} 个采样点</span>
              <span class="spacer" />
              <button class="chip" :class="{ 'is-on': chartTable }" @click="chartTable = !chartTable">表格</button>
            </div>
            <LossChart :points="points" :height="190" :show-table="chartTable" />
          </div>

          <section class="samples" aria-label="最新采样">
            <div class="row samples__head">
              <span class="label-pill">最新采样</span>
              <span v-if="activeGroup" class="muted num">{{ activeGroup.label }}</span>
              <span class="spacer" />
              <RouterLink
                v-if="activeImages.length"
                :to="{ name: 'results', params: { taskId: task.task_id } }"
                class="chip"
              >全部</RouterLink>
            </div>

            <button
              v-if="heroImage"
              class="samples__stage"
              :title="`${heroImage.file_name}（点击放大）`"
              @click="viewerIndex = 0"
            >
              <img v-if="heroImage.preview_url" :src="heroImage.preview_url" :alt="heroImage.file_name" />
              <span v-else class="samples__empty">预览不可用</span>
            </button>
            <div v-else class="samples__stage samples__stage--empty">
              <span class="samples__empty">
                还没有采样图
                <small>引擎按 SAMPLE_EVERY 在训练进程内生成</small>
              </span>
            </div>

            <div v-if="sampleGroups.length" class="samples__steps">
              <button
                v-for="g in sampleGroups"
                :key="g.key"
                class="chip"
                :class="{ 'is-on': activeGroup?.key === g.key }"
                :title="pinnedStep === g.key ? '点击回到最新' : `查看 ${g.label}`"
                @click="pinStep(g.key)"
              >{{ g.label }}</button>
            </div>

            <div v-if="activeImages.length > 1" class="samples__thumbs">
              <button
                v-for="(a, i) in activeImages"
                :key="a.artifact_id"
                class="samples__thumb"
                :title="a.file_name"
                @click="viewerIndex = i"
              >
                <img v-if="a.preview_url" :src="a.preview_url" :alt="a.file_name" loading="lazy" />
              </button>
            </div>

            <div v-if="prompts.length || sampleSettings" class="samples__prompt">
              <p v-if="sampleSettings" class="samples__meta">{{ sampleSettings }}</p>
              <div v-for="(p, i) in prompts.slice(0, 2)" :key="i" class="samples__pline">
                <span class="samples__pidx">{{ i + 1 }}</span>
                <span class="samples__ptext" :title="p">{{ p }}</span>
                <button class="chip" @click="copyText(p, '提示词')">复制</button>
              </div>
              <p v-if="seeds.length" class="samples__meta">seed {{ seeds.join(', ') }}</p>
            </div>
          </section>
        </div>
      </div>

      <div class="detail__bottom">
        <!-- Log -->
        <section class="log panel">
          <div class="log__head">
            <span class="label-pill"><AppIcon name="terminal" :size="12" />stdout.log</span>
            <span class="muted num">尾部 {{ lines.length }} 行</span>
            <span class="spacer" />
            <div class="log__search">
              <AppIcon name="search" :size="13" />
              <input v-model="logQuery" placeholder="搜索日志" spellcheck="false" />
            </div>
            <button class="chip" :class="{ 'is-on': follow }" @click="follow = !follow">跟随</button>
          </div>
          <div ref="logEl" class="log__body scroll selectable" @scroll="onLogScroll">
            <div v-for="(l, i) in shownLines" :key="i" class="log__line" :class="lineClass(l)">{{ l }}</div>
            <div v-if="!shownLines.length" class="muted log__empty">{{ logQuery ? '无匹配行' : '暂无日志' }}</div>
          </div>
        </section>

        <!-- Info -->
        <section class="info panel panel--pad stack">
          <span class="label-pill" style="align-self: flex-start">任务信息</span>
          <dl class="kv selectable">
            <dt>数据集</dt>
            <dd>{{ task.dataset_name ?? '—' }}</dd>
            <dt>创建</dt>
            <dd>{{ fmtTime(task.created_at) }}</dd>
            <dt>开始</dt>
            <dd>{{ fmtTime(task.started_at) }}</dd>
            <dt>结束</dt>
            <dd>{{ fmtTime(task.finished_at) }}</dd>
            <dt>输出</dt>
            <dd class="mono" style="font-size: 11px">{{ task.output_dir }}</dd>
          </dl>
          <div class="caps">
            <span :class="{ on: task.recovery.graceful_stop }">正常停止</span>
            <span :class="{ on: task.recovery.resume_from_weights }">权重续训</span>
            <span :class="{ on: task.recovery.resume_full_state }">完整状态恢复</span>
          </div>
          <div class="row" style="margin-top: auto; flex-wrap: wrap">
            <button class="btn btn--sm" @click="reveal"><AppIcon name="folder" :size="13" />打开输出目录</button>
            <RouterLink v-if="['succeeded', 'stopped', 'running'].includes(task.state)" :to="{ name: 'results', params: { taskId: task.task_id } }" class="btn btn--sm">
              查看产物
            </RouterLink>
          </div>
        </section>
      </div>
    </section>

    <section v-else class="tasks__detail panel">
      <EmptyState
        icon="activity"
        :title="store.loadError ? `加载失败：${store.loadError}` : store.tasksLoaded ? '还没有训练任务' : '正在读取任务…'"
        :text="store.loadError ? '任务列表未能从本地后端读取；恢复连接后会自动重试。' : '新建训练并提交后，任务会在这里排队和运行。'"
      >
        <RouterLink :to="{ name: 'train' }" class="btn btn--primary">新建训练</RouterLink>
      </EmptyState>
    </section>

    <Teleport to="body">
      <div v-if="menu" ref="menuEl" class="task-menu" role="menu" aria-label="任务操作"
        :style="{ left: `${menu.x}px`, top: `${menu.y}px` }" @contextmenu.prevent>
        <button role="menuitem" :disabled="historyBusy" @click="historyAction('export', menu.target)">导出配置</button>
        <button role="menuitem" :disabled="historyBusy" @click="historyAction('use', menu.target)">使用配置</button>
        <button role="menuitem" class="task-menu__delete" :disabled="historyBusy || !deletable(menu.target)"
          :title="deletable(menu.target) ? '删除历史任务' : '请先结束任务；连接丢失时需确认进程已退出'"
          @click="historyAction('delete', menu.target)">删除</button>
      </div>
    </Teleport>
    <ImageViewer v-model:index="viewerIndex" :items="viewerItems" />
    <TrainingExportDialog v-if="exportSource" :source="exportSource" @close="exportSource = null" />
    <ModalDialog v-if="deleteTarget" title="删除历史训练任务？" @close="!historyBusy && (deleteTarget = null)">
      <p>删除 <strong>{{ deleteTarget.name }}</strong> 的任务记录、配置快照、运行日志和指标。</p>
      <p class="muted">不会删除数据集、底模、引擎环境、已生成的模型或预览图。此操作不可撤销，建议先导出配置。</p>
      <template #footer>
        <button class="btn btn--ghost" :disabled="historyBusy" @click="deleteTarget = null">取消</button>
        <button class="btn btn--danger" :disabled="historyBusy" @click="deleteTask">{{ historyBusy ? '正在删除…' : '确认删除' }}</button>
      </template>
    </ModalDialog>
    <ModalDialog v-if="confirmKind && task" :title="task.state === 'connection_lost' ? '确认原训练进程已退出？' : confirmKind === 'kill' ? '强制结束训练进程？' : '请求中断训练？'" @close="confirmKind = null">
      <template v-if="confirmKind === 'kill'">
        <p>将结束 <strong>{{ task.name }}</strong> 的整个训练进程树。</p>
        <p class="warnbox">
          <AppIcon name="alert" :size="14" />
          如果进程正在写入 checkpoint，文件可能损坏或不完整；未完成的文件不会被登记为可用产物。
        </p>
      </template>
      <template v-else-if="task.state === 'connection_lost'">
        <p>请先在系统任务管理器中核实 <strong>{{ task.name }}</strong> 的原训练进程和子进程均已退出。</p>
        <p class="warnbox">此操作只记录你的人工确认，不会结束未知进程；确认后将解除队列阻塞。原进程仍在运行时不要确认。</p>
      </template>
      <template v-else>
        <p>向 <strong>{{ task.name }}</strong> 发送中断请求；排队任务直接取消。进程未及时退出时会强制结束。</p>
        <p class="muted">不保证停止时一定生成完整 checkpoint；是否可继续取决于引擎声明的恢复能力。</p>
      </template>
      <template #footer>
        <button class="btn btn--ghost" @click="confirmKind = null">取消</button>
        <button class="btn" :class="confirmKind === 'kill' ? 'btn--danger' : 'btn--primary'" :disabled="acting" @click="doStop">
          {{ task.state === 'connection_lost' ? '我已确认进程退出' : confirmKind === 'kill' ? '强制结束' : '确认中断' }}
        </button>
      </template>
    </ModalDialog>
  </div>
</template>

<style scoped>
.task-menu { position: fixed; z-index: 10000; min-width: 168px; padding: 6px; border: 1px solid var(--line); border-radius: var(--radius-md); background: var(--panel-raised); box-shadow: 0 12px 36px rgba(0,0,0,.35); animation: fade-up .16s var(--ease); }
.task-menu button { display: block; width: 100%; padding: 10px 12px; border-radius: var(--radius-sm); text-align: left; color: var(--text); background: transparent; }
.task-menu button:hover:not(:disabled), .task-menu button:focus-visible { background: rgba(255,255,255,.06); outline: 1px solid var(--line); }
.task-menu button:disabled { opacity: .4; cursor: not-allowed; }
.task-menu .task-menu__delete { color: var(--danger); }

.tasks {
  display: grid;
  grid-template-columns: 290px minmax(0, 1fr);
  gap: var(--gap);
  height: 100%;
}
.tasks__list {
  display: flex;
  flex-direction: column;
  min-height: 0;
}
.list__head {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 20px 16px 10px 40px;
}
.list__filter {
  padding: 0 16px 12px 40px;
}
.list__filter :deep(.chip) {
  height: 24px;
  padding: 0 10px;
  font-size: 11px;
}
.list__body {
  flex: 1;
  padding: 0 10px 12px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.list__empty {
  padding: 12px 30px;
}
.titem {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 12px 14px;
  border-radius: var(--radius-md);
  text-align: left;
  transition: background 0.15s;
}
.titem:hover {
  background: rgba(255, 255, 255, 0.05);
}
.titem.is-on {
  background: var(--light);
  color: var(--ink);
}
.titem.is-on :deep(.tag) {
  background: rgba(0, 0, 0, 0.1);
  color: var(--ink);
}
.titem__meta {
  font-size: 11px;
  opacity: 0.6;
}

.tasks__detail {
  display: flex;
  flex-direction: column;
  gap: var(--gap);
  min-height: 0;
  min-width: 0;
}
.detail__top {
  display: flex;
  flex-direction: column;
  gap: 16px;
  padding: 22px 26px 18px;
  border-radius: var(--radius-xl);
}
.detail__head {
  display: flex;
  align-items: center;
  gap: 10px;
}
.err {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 14px;
  border-radius: var(--radius-sm);
  background: var(--danger-bg);
  color: var(--danger);
}
.metrics {
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: 12px;
}
.m {
  display: flex;
  flex-direction: column;
}
.m__l {
  font-size: 11px;
  color: var(--text-3);
}
.m__v {
  font-size: 19px;
  font-weight: 600;
  white-space: nowrap;
}
.m__v small {
  font-size: 11px;
  font-weight: 400;
  color: var(--text-3);
}
.chart-wrap {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

/* Loss and the newest sample share one screen; the engine's own numbers sit above. */
.detail__grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 300px;
  gap: 16px;
  align-items: start;
}
.runtime {
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: 10px;
  padding-top: 12px;
  border-top: 1px solid var(--line);
}
.rt {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
}
.rt__l {
  font-size: 10.5px;
  color: var(--text-3);
}
.rt__v {
  font-size: 12.5px;
  font-weight: 600;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.rt__v small {
  display: block;
  font-size: 10.5px;
  font-weight: 400;
  color: var(--text-3);
  overflow: hidden;
  text-overflow: ellipsis;
}
.samples {
  display: flex;
  flex-direction: column;
  gap: 8px;
  min-width: 0;
  padding: 10px;
  border-radius: var(--radius-md);
  background: var(--field);
}
.samples__head {
  gap: 8px;
}
.samples__head .chip {
  height: 22px;
  font-size: 11px;
}
.samples__stage {
  display: grid;
  place-items: center;
  width: 100%;
  height: 148px;
  padding: 0;
  border: 0;
  border-radius: var(--radius-sm);
  overflow: hidden;
  background: var(--code-bg);
}
.samples__stage img {
  width: 100%;
  height: 100%;
  object-fit: contain;
}
.samples__stage--empty {
  border: 1px dashed var(--line-strong);
  background: transparent;
}
.samples__empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  padding: 0 12px;
  font-size: 11.5px;
  color: var(--text-3);
  text-align: center;
}
.samples__empty small {
  font-size: 10.5px;
  opacity: 0.8;
}
.samples__steps,
.samples__thumbs {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.samples__steps .chip {
  height: 22px;
  font-size: 11px;
}
.samples__thumb {
  width: 34px;
  height: 34px;
  padding: 0;
  border-radius: 6px;
  overflow: hidden;
  background: var(--code-bg);
}
.samples__thumb img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}
.samples__prompt {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
}
.samples__meta {
  margin: 0;
  font-size: 10.5px;
  color: var(--text-3);
}
.samples__pline {
  display: flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
}
.samples__pidx {
  flex: none;
  width: 14px;
  font-size: 10px;
  color: var(--text-3);
}
.samples__ptext {
  flex: 1;
  min-width: 0;
  font-size: 11px;
  color: var(--text-2);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.samples__pline .chip {
  flex: none;
  height: 22px;
  font-size: 10.5px;
}
.chart-head .chip {
  height: 24px;
  font-size: 11px;
}

.detail__bottom {
  flex: 1;
  min-height: 0;
  display: grid;
  grid-template-columns: minmax(0, 1fr) 280px;
  gap: var(--gap);
}
.log {
  display: flex;
  flex-direction: column;
  min-height: 0;
}
.log__head {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 14px 16px 10px;
}
.log__head .chip {
  height: 26px;
}
.log__search {
  display: flex;
  align-items: center;
  gap: 6px;
  height: 28px;
  padding: 0 12px;
  border-radius: var(--radius-pill);
  background: var(--field);
  color: var(--text-3);
}
.log__search input {
  width: 120px;
  border: 0;
  outline: 0;
  background: none;
  font-size: 12px;
}
.log__body {
  flex: 1;
  min-height: 120px;
  margin: 0 12px 12px;
  padding: 12px 14px;
  border-radius: var(--radius-md);
  background: var(--code-bg);
  font-family: var(--font-mono);
  font-size: 11.5px;
  line-height: 1.65;
  color: var(--text-2);
}
.log__line {
  white-space: pre-wrap;
  word-break: break-all;
}
.log__line.is-err {
  color: var(--danger);
}
.log__line.is-warn {
  color: var(--warn);
}
.log__line.is-sys {
  color: var(--info);
}
.log__empty {
  font-family: var(--font-sans);
}
.info {
  min-height: 0;
  overflow: auto;
}
.caps {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.caps span {
  padding: 3px 10px;
  border-radius: var(--radius-pill);
  border: 1px dashed var(--line-strong);
  font-size: 11px;
  color: var(--text-3);
  text-decoration: line-through;
}
.caps span.on {
  border-style: solid;
  color: var(--text-2);
  text-decoration: none;
}
.warnbox {
  display: flex;
  gap: 8px;
  padding: 10px 12px;
  border-radius: var(--radius-sm);
  background: var(--warn-bg);
  color: var(--warn);
  font-size: 12px;
}
.warnbox .icon {
  margin-top: 2px;
}

@media (max-width: 1180px) {
  .tasks {
    grid-template-columns: 240px minmax(0, 1fr);
  }
  .metrics {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }
  .detail__grid {
    grid-template-columns: minmax(0, 1fr);
  }
  .runtime {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }
  .detail__bottom {
    grid-template-columns: minmax(0, 1fr);
  }
  .info {
    display: none;
  }
}
</style>
