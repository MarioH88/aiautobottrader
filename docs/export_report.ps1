# Opens the completion report in your default browser for PDF export.
# In the browser: Ctrl+P -> Save as PDF
$report = Join-Path $PSScriptRoot "PROJECT_COMPLETION_REPORT.html"
Start-Process $report
Write-Host "Report opened. Use Print -> Save as PDF to upload."
