import { tool } from "@opencode-ai/plugin"

export const top = tool({
  description: "Show top N trading strategies by Sharpe ratio from the strategy registry. Use when user asks about best strategies, top performers, champions.",
  args: {
    n: tool.schema.number().default(10).describe("Number of top strategies to show"),
    minSh: tool.schema.number().optional().describe("Minimum Sharpe filter"),
    maxDd: tool.schema.number().optional().describe("Maximum drawdown filter"),
  },
  async execute(args) {
    let cmd = `python3 -m src.strategies.registry --top ${args.n}`
    if (args.minSh !== undefined) cmd += ` --min-sharpe ${args.minSh}`
    if (args.maxDd !== undefined) cmd += ` --max-dd ${args.maxDd}`
    return (await Bun.$`${cmd}`.text()).trim()
  },
})

export const lookup = tool({
  description: "Look up a trading strategy by name or partial name in the strategy registry. Use when user asks about a specific strategy, its metrics, or to find strategies by keyword.",
  args: {
    name: tool.schema.string().describe("Strategy name or partial name to search for"),
  },
  async execute(args) {
    return (await Bun.$`python3 -m src.strategies.registry --lookup ${args.name}`.text()).trim()
  },
})

export const stats = tool({
  description: "Show strategy registry statistics: total strategies, filters, champion metrics, date range. Use when user asks about registry health, coverage, or summary.",
  args: {},
  async execute() {
    return (await Bun.$`python3 -m src.strategies.registry --stats`.text()).trim()
  },
})

export const discover = tool({
  description: "Scan the verified/ strategies directory and report which champion strategies are available. Use when user asks what verified strategies exist.",
  args: {},
  async execute() {
    return (await Bun.$`python3 -m src.strategies.registry --discover`.text()).trim()
  },
})
