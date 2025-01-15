interface DateFormatterProps {
  date: string
}

export function DateFormatter({ date }: DateFormatterProps) {
  const formattedDate = new Date(date).toLocaleString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })

  return <span className="text-sm text-gray-400">{formattedDate}</span>
}

