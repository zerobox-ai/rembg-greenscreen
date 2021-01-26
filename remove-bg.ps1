param([int]$batches,
[int]$concurrency,
[string]$target,
[int]$fps)

ffmpeg -i $target -vf fps=$fps %d.jpg

$folder = Get-Location

# delete any failed png conversions
ls *.png | where Length -eq 0 | del

$d = 0;

Get-ChildItem | 
    ForEach-Object { 
        New-Object PSObject  -property @{
            Test= -Not (Test-Path -Path ($_.name -replace "jpg", ".out.png") ); 
            Path=$_.Name;
            Number=$d++
        } } | 
        Where-Object Test | 
        Group-Object -Property {$_.Number % $batches} |
        ForEach-Object{
            $gn = $_.Name
            mkdir $gn -Force

            $_.Group | ForEach-Object{ Copy-Item $_.Path $gn }

            Start-ThreadJob -ThrottleLimit $concurrency -ScriptBlock {param([string]$folder, [string]$gn)

                C:/Windows/System32/cmd.exe /K "$env:anaconda\\Scripts\\activate.bat $env:anaconda && cd $env:rembg && python -m src.rembg.cmd.cli -p $folder\$gn 2>&1 && exit"
            
                Copy-Item $folder/$gn/*.png $folder
                Remove-Item $folder/$gn/ -Force -Recurse

            } -ArgumentList "$folder", "$gn"

        }

get-job | Wait-Job

# do a second pass to pick up the failed ones
$failed = ls | where Length -eq 0 |  %{ $_.Name -replace ".out.png", "" }

mkdir failed -Force
$failed | ForEach-Object{ Copy-Item "$_.jpg" failed }

C:/Windows/System32/cmd.exe /K "$env:anaconda\\Scripts\\activate.bat $env:anaconda && cd $env:rembg && python -m src.rembg.cmd.cli -p $folder\failed 2>&1 && exit"

Copy-Item $folder/failed/*.png $folder
Remove-Item $folder/failed/ -Force -Recurse

# if no failed pngs and >100 non failed pngs
if( (ls *.png | where Length -eq 0).Count -eq 0 `
        -and (ls *.png | where Length -gt 0).Count -gt 100)
{
    del *.jpg
    
    ffmpeg -i %d.out.png -vcodec png ($target -replace ".mp4", ".mov")

    del *.png
} 